use std::ffi::NulError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("string contains an interior NUL byte")]
    Nul(#[from] NulError),
    #[error("{0}")]
    Native(String),
    #[error("buffer size does not match {width}x{height} RGBA8 image")]
    InvalidRgbaBuffer { width: u32, height: u32 },
    #[error("image dimensions must be positive, got {width}x{height}")]
    InvalidImageDimensions { width: u32, height: u32 },
    #[error("failed to allocate image buffer for {width}x{height} with {channels} channels")]
    ImageAllocationFailed {
        width: u32,
        height: u32,
        channels: usize,
    },
    #[error(
        "image dimensions {width}x{height} with {channels} channels exceed the addressable buffer size"
    )]
    ImageSizeOverflow {
        width: u32,
        height: u32,
        channels: usize,
    },
}

pub type Result<T> = std::result::Result<T, Error>;
