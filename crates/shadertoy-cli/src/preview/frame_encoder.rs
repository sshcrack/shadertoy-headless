use super::{EffectivePreviewTransport, PreviewStatus};
use crate::ops::{preview_jpeg_bytes, preview_png_bytes, preview_raw_rgb_bytes};
use anyhow::{Context, Result, bail};
use axum::body::Bytes;
use shadertoy::RgbImage;
use std::io::{BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use tokio::sync::broadcast;

#[derive(Default)]
struct EncoderState {
    pending: Option<RgbImage>,
    shutdown: bool,
}

#[derive(Clone)]
pub(super) struct FrameSubmitter {
    state: Arc<(Mutex<EncoderState>, Condvar)>,
}

impl FrameSubmitter {
    pub(super) fn submit(&self, image: RgbImage) {
        let (lock, wake) = &*self.state;
        let mut state = lock.lock().expect("preview encoder lock poisoned");
        state.pending = Some(image);
        wake.notify_one();
    }
}

pub(super) struct FrameEncoder {
    submitter: FrameSubmitter,
    worker: Option<JoinHandle<()>>,
}

impl FrameEncoder {
    pub(super) fn start(
        frame_image: Arc<RwLock<Bytes>>,
        status: Arc<RwLock<PreviewStatus>>,
        updates: broadcast::Sender<String>,
        frames: broadcast::Sender<Bytes>,
        transport: EffectivePreviewTransport,
    ) -> Result<Self> {
        let state = Arc::new((Mutex::new(EncoderState::default()), Condvar::new()));
        let submitter = FrameSubmitter {
            state: Arc::clone(&state),
        };
        let worker = thread::Builder::new()
            .name("shadertoy-preview-encode".into())
            .spawn(move || encode_loop(state, frame_image, status, updates, frames, transport))
            .context("failed to start preview frame encoder")?;

        Ok(Self {
            submitter,
            worker: Some(worker),
        })
    }

    pub(super) fn submitter(&self) -> FrameSubmitter {
        self.submitter.clone()
    }
}

impl Drop for FrameEncoder {
    fn drop(&mut self) {
        let (lock, wake) = &*self.submitter.state;
        {
            let mut state = lock.lock().expect("preview encoder lock poisoned");
            state.shutdown = true;
            state.pending = None;
            wake.notify_one();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct MjpegEncoder {
    child: Child,
    stdin: ChildStdin,
    output_rx: Receiver<Result<Vec<u8>>>,
    reader: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
}

impl MjpegEncoder {
    fn start(width: u32, height: u32) -> Result<Self> {
        let executable = std::env::var_os("SHADERTOY_FFMPEG").unwrap_or_else(|| "ffmpeg".into());
        let mut child = Command::new(executable)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                &format!("{width}x{height}"),
                "-i",
                "pipe:0",
                "-an",
                "-vf",
                "vflip",
                "-c:v",
                "mjpeg",
                "-threads",
                "1",
                "-q:v",
                "3",
                "-flush_packets",
                "1",
                "-f",
                "mjpeg",
                "pipe:1",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to launch ffmpeg preview encoder")?;
        let stdin = child
            .stdin
            .take()
            .context("failed to open ffmpeg preview stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("failed to open ffmpeg preview stdout")?;
        let (output_tx, output_rx) = mpsc::channel();
        let reader = thread::Builder::new()
            .name("shadertoy-preview-mjpeg-read".into())
            .spawn(move || {
                let mut stdout = BufReader::new(stdout);
                loop {
                    let frame = read_jpeg_frame(&mut stdout);
                    let stop = frame.is_err();
                    if output_tx.send(frame).is_err() || stop {
                        break;
                    }
                }
            })
            .context("failed to start ffmpeg preview reader")?;
        Ok(Self {
            child,
            stdin,
            output_rx,
            reader: Some(reader),
            width,
            height,
        })
    }

    fn matches(&self, image: &RgbImage) -> bool {
        self.width == image.width && self.height == image.height
    }

    fn encode(&mut self, image: &RgbImage) -> Result<Vec<u8>> {
        if !self.matches(image) {
            bail!("preview encoder dimensions changed");
        }
        self.stdin
            .write_all(&image.pixels)
            .context("ffmpeg stopped accepting preview frames")?;
        self.output_rx
            .recv()
            .context("ffmpeg preview reader stopped")?
    }
}

impl Drop for MjpegEncoder {
    fn drop(&mut self) {
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

fn read_jpeg_frame(reader: &mut impl Read) -> Result<Vec<u8>> {
    let mut frame = Vec::with_capacity(256 * 1024);
    let mut previous = 0u8;
    loop {
        let mut byte = [0u8; 1];
        reader
            .read_exact(&mut byte)
            .context("ffmpeg stopped producing preview frames")?;
        frame.push(byte[0]);
        if previous == 0xff && byte[0] == 0xd9 {
            return Ok(frame);
        }
        previous = byte[0];
    }
}

fn encode_loop(
    state: Arc<(Mutex<EncoderState>, Condvar)>,
    frame_image: Arc<RwLock<Bytes>>,
    status: Arc<RwLock<PreviewStatus>>,
    updates: broadcast::Sender<String>,
    frames: broadcast::Sender<Bytes>,
    transport: EffectivePreviewTransport,
) {
    let mut mjpeg: Option<MjpegEncoder> = None;
    let mut ffmpeg_disabled = false;

    loop {
        let image = {
            let (lock, wake) = &*state;
            let mut state = lock.lock().expect("preview encoder lock poisoned");
            while state.pending.is_none() && !state.shutdown {
                state = wake.wait(state).expect("preview encoder lock poisoned");
            }
            if state.shutdown {
                return;
            }
            state.pending.take()
        };

        let Some(image) = image else {
            continue;
        };

        if mjpeg
            .as_ref()
            .is_some_and(|encoder| !encoder.matches(&image))
        {
            mjpeg = None;
        }

        let encoded = match transport {
            EffectivePreviewTransport::Raw => preview_raw_rgb_bytes(&image),
            EffectivePreviewTransport::Png => preview_png_bytes(&image),
            EffectivePreviewTransport::Mjpeg => {
                if ffmpeg_disabled {
                    preview_jpeg_bytes(&image)
                } else {
                    if mjpeg.is_none() {
                        match MjpegEncoder::start(image.width, image.height) {
                            Ok(encoder) => mjpeg = Some(encoder),
                            Err(_) => ffmpeg_disabled = true,
                        }
                    }
                    match mjpeg.as_mut() {
                        Some(encoder) => match encoder.encode(&image) {
                            Ok(frame) => Ok(frame),
                            Err(_) => {
                                mjpeg = None;
                                ffmpeg_disabled = true;
                                preview_jpeg_bytes(&image)
                            }
                        },
                        None => preview_jpeg_bytes(&image),
                    }
                }
            }
        };

        match encoded {
            Ok(encoded) => {
                let encoded = Bytes::from(encoded);
                *frame_image.write().expect("preview frame lock poisoned") = encoded.clone();
                let _ = frames.send(encoded);
            }
            Err(error) => {
                let mut status = status.write().expect("preview status lock poisoned");
                status.error = Some(error.to_string());
                if let Ok(message) = serde_json::to_string(&*status) {
                    let _ = updates.send(message);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn image(marker: u8) -> RgbImage {
        RgbImage::new(1, 1, vec![marker, marker, marker])
    }

    #[test]
    fn pending_slot_keeps_only_the_latest_frame() {
        let state = Arc::new((Mutex::new(EncoderState::default()), Condvar::new()));
        let submitter = FrameSubmitter {
            state: Arc::clone(&state),
        };

        submitter.submit(image(1));
        submitter.submit(image(2));

        let pending = state.0.lock().unwrap().pending.take().unwrap();
        assert_eq!(pending.pixels, vec![2, 2, 2]);
    }

    #[test]
    fn jpeg_stream_reader_stops_at_end_marker() {
        let mut reader = Cursor::new(vec![0xff, 0xd8, 1, 2, 0xff, 0xd9, 9, 9]);
        assert_eq!(
            read_jpeg_frame(&mut reader).unwrap(),
            vec![0xff, 0xd8, 1, 2, 0xff, 0xd9]
        );
        assert_eq!(reader.position(), 6);
    }
}
