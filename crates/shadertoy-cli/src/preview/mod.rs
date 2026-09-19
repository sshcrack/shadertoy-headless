use crate::manifest::{LoadedManifest, PassKind};
use crate::ops::rgb_png_bytes;
use crate::project::{build_native_project, ensure_source_files_exist};
use anyhow::{Context, Result, bail};
use axum::extract::{
    Query, State,
    ws::{Message, WebSocket, WebSocketUpgrade},
};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use futures_util::StreamExt;
use notify::{RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use shadertoy::{HeadlessContext, Runtime};
use std::collections::BTreeMap;
use std::io::IsTerminal;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

mod renderer;
mod state;
mod web;

use renderer::render_loop;
use state::{relevant_watch_path, reload_event_kind};
use web::{frame_png, index, status, validate_remote_auth, websocket};

#[derive(Debug, Clone)]
pub struct PreviewConfig {
    pub project: PathBuf,
    pub host: String,
    pub port: u16,
    pub open: bool,
    pub no_open: bool,
    pub token: Option<String>,
    pub preserve_reload_state: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct PreviewStatus {
    pub project: String,
    pub sequence: u64,
    pub frame: i32,
    pub time: f32,
    pub paused: bool,
    pub time_scale: f32,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub view: String,
    pub final_pass: String,
    pub passes: Vec<String>,
    pub error: Option<String>,
}

impl Default for PreviewStatus {
    fn default() -> Self {
        Self {
            project: String::new(),
            sequence: 0,
            frame: 0,
            time: 0.0,
            paused: false,
            time_scale: 0.0,
            width: 1280,
            height: 720,
            fps: 60.0,
            view: "image".into(),
            final_pass: "image".into(),
            passes: Vec::new(),
            error: None,
        }
    }
}

#[derive(Clone)]
struct Shared {
    frame_png: Arc<RwLock<Vec<u8>>>,
    status: Arc<RwLock<PreviewStatus>>,
    controls: mpsc::Sender<Control>,
    updates: broadcast::Sender<String>,
    clients: Arc<AtomicUsize>,
    token: Option<Arc<String>>,
}

#[derive(Debug)]
enum Control {
    Reload,
    Pause,
    Resume,
    Reset,
    Step,
    View(String),
    Resolution(u32, u32),
    TimeScale(f32),
    Mouse {
        x: f32,
        y: f32,
        down: bool,
        clicked: bool,
    },
    Key {
        code: u8,
        down: bool,
        pressed: bool,
    },
    Shutdown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum BrowserControl {
    Pause,
    Resume,
    Reset,
    Step,
    View {
        pass: String,
    },
    Resolution {
        width: u32,
        height: u32,
    },
    TimeScale {
        value: f32,
    },
    Mouse {
        x: f32,
        y: f32,
        down: bool,
        #[serde(default)]
        clicked: bool,
    },
    Key {
        code: u8,
        down: bool,
        #[serde(default)]
        pressed: bool,
    },
}

#[derive(Debug, Deserialize, Default)]
struct AuthQuery {
    token: Option<String>,
}

struct SavedRuntimeState {
    time: f32,
    frame: i32,
    fps: f32,
    buffers: BTreeMap<String, Vec<f32>>,
}
pub async fn run(config: PreviewConfig, json_mode: bool) -> Result<()> {
    if config.open && config.no_open {
        bail!("--open and --no-open are mutually exclusive");
    }
    validate_remote_auth(&config)?;

    let loaded = LoadedManifest::load(&config.project)?;
    let root = loaded.root.clone();
    let initial_status = PreviewStatus {
        project: loaded.manifest.project.name.clone(),
        width: loaded.manifest.render.width,
        height: loaded.manifest.render.height,
        fps: loaded.manifest.render.fps,
        final_pass: loaded.manifest.final_pass().name.clone(),
        view: loaded.manifest.final_pass().name.clone(),
        passes: loaded
            .manifest
            .passes
            .iter()
            .map(|pass| pass.name.clone())
            .collect(),
        ..PreviewStatus::default()
    };

    let (control_tx, control_rx) = mpsc::channel();
    let (update_tx, _) = broadcast::channel(128);
    let shared = Shared {
        frame_png: Arc::new(RwLock::new(Vec::new())),
        status: Arc::new(RwLock::new(initial_status)),
        controls: control_tx.clone(),
        updates: update_tx.clone(),
        clients: Arc::new(AtomicUsize::new(0)),
        token: config.token.clone().map(Arc::new),
    };

    let (startup_tx, startup_rx) = mpsc::sync_channel(1);
    let render_shared = shared.clone();
    let preserve_reload_state = config.preserve_reload_state;
    let render_root = root.clone();
    let render_thread = thread::spawn(move || {
        render_loop(
            render_root,
            render_shared,
            control_rx,
            startup_tx,
            preserve_reload_state,
        );
    });

    match startup_rx.recv_timeout(Duration::from_secs(15)) {
        Ok(Ok(())) => {}
        Ok(Err(message)) => {
            let _ = control_tx.send(Control::Shutdown);
            let _ = render_thread.join();
            bail!("preview renderer failed to start: {message}");
        }
        Err(error) => {
            let _ = control_tx.send(Control::Shutdown);
            let _ = render_thread.join();
            bail!("preview renderer did not start: {error}");
        }
    }

    let watcher_tx = control_tx.clone();
    let watch_root = root.clone();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        let Ok(event) = event else {
            return;
        };
        if reload_event_kind(&event.kind)
            && event
                .paths
                .iter()
                .any(|path| relevant_watch_path(&watch_root, path))
        {
            let _ = watcher_tx.send(Control::Reload);
        }
    })
    .context("failed to create project file watcher")?;
    watcher
        .watch(&root, RecursiveMode::Recursive)
        .with_context(|| format!("failed to watch {}", root.display()))?;

    let app = Router::new()
        .route("/", get(index))
        .route("/frame.png", get(frame_png))
        .route("/api/status", get(status))
        .route("/ws", get(websocket))
        .with_state(shared.clone());

    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port))
        .await
        .with_context(|| {
            format!(
                "failed to bind preview server on {}:{}",
                config.host, config.port
            )
        })?;
    let address = listener.local_addr()?;
    let query = config
        .token
        .as_ref()
        .map(|token| format!("?token={token}"))
        .unwrap_or_default();
    let display_host = if config.host == "0.0.0.0" || config.host == "::" {
        "127.0.0.1"
    } else {
        config.host.as_str()
    };
    let url = format!("http://{}:{}{}", display_host, address.port(), query);

    if json_mode {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "ok": true,
                "action": "preview",
                "url": url,
                "host": config.host,
                "port": address.port(),
                "project": root,
            }))?
        );
    } else {
        println!("ShaderToy preview: {url}");
        println!("Watching {}", root.display());
    }

    let should_open = config.open
        || (!config.no_open && std::io::stdout().is_terminal() && config.host != "0.0.0.0");
    if should_open && let Err(error) = open::that(&url) {
        eprintln!("warning: could not open browser: {error}");
    }

    let server_result = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await;

    let _ = control_tx.send(Control::Shutdown);
    drop(watcher);
    let _ = render_thread.join();

    server_result.context("preview server failed")
}
