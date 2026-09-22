use super::*;

pub(super) fn validate_remote_auth(config: &PreviewConfig) -> Result<()> {
    let loopback = config.host == "localhost"
        || config
            .host
            .parse::<IpAddr>()
            .map(|address| address.is_loopback())
            .unwrap_or(false);
    if !loopback && config.token.as_deref().unwrap_or_default().is_empty() {
        bail!(
            "non-loopback preview binding requires --token; bind to 127.0.0.1 for local-only preview"
        );
    }
    Ok(())
}

fn authorized(shared: &Shared, query: &AuthQuery) -> bool {
    match &shared.token {
        None => true,
        Some(expected) => query.token.as_deref() == Some(expected.as_str()),
    }
}

pub(super) async fn index(
    State(shared): State<Shared>,
    Query(query): Query<AuthQuery>,
) -> Response {
    if !authorized(&shared, &query) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    Html(INDEX_HTML).into_response()
}

pub(super) async fn frame_png(
    State(shared): State<Shared>,
    Query(query): Query<AuthQuery>,
) -> Response {
    if !authorized(&shared, &query) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let frame = shared
        .frame_png
        .read()
        .expect("preview frame lock poisoned")
        .clone();
    if frame.is_empty() {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    (StatusCode::OK, headers, frame).into_response()
}

pub(super) async fn status(
    State(shared): State<Shared>,
    Query(query): Query<AuthQuery>,
) -> Response {
    if !authorized(&shared, &query) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let snapshot = shared
        .status
        .read()
        .expect("preview status lock poisoned")
        .clone();
    Json(snapshot).into_response()
}

pub(super) async fn websocket(
    ws: WebSocketUpgrade,
    State(shared): State<Shared>,
    Query(query): Query<AuthQuery>,
) -> Response {
    if !authorized(&shared, &query) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    ws.on_upgrade(move |socket| websocket_loop(socket, shared))
}

async fn websocket_loop(mut socket: WebSocket, shared: Shared) {
    shared.clients.fetch_add(1, Ordering::Relaxed);
    let mut updates = shared.updates.subscribe();
    let mut frames = shared.frames.subscribe();

    let initial = shared
        .status
        .read()
        .expect("preview status lock poisoned")
        .clone();
    if let Ok(message) = serde_json::to_string(&initial) {
        let _ = socket.send(Message::Text(message.into())).await;
    }
    let initial_frame = shared
        .frame_png
        .read()
        .expect("preview frame lock poisoned")
        .clone();
    if !initial_frame.is_empty() {
        let _ = socket.send(Message::Binary(initial_frame)).await;
    }

    loop {
        tokio::select! {
            incoming = socket.next() => {
                let Some(Ok(message)) = incoming else { break; };
                match message {
                    Message::Text(text) => {
                        if let Ok(command) = serde_json::from_str::<BrowserControl>(&text) {
                            let _ = shared.controls.send(browser_control(command));
                        }
                    }
                    Message::Binary(data) => {
                        let expected = (crate::media::WEBCAM_WIDTH as usize)
                            * (crate::media::WEBCAM_HEIGHT as usize)
                            * 4;
                        let webcam_enabled = shared
                            .status
                            .read()
                            .map(|status| status.webcam)
                            .unwrap_or(false);
                        if webcam_enabled && data.len() == expected {
                            let _ = shared.controls.send(Control::WebcamFrame(data.to_vec()));
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            update = updates.recv() => {
                match update {
                    Ok(update) => {
                        if socket.send(Message::Text(update.into())).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            frame = frames.recv() => {
                match frame {
                    Ok(frame) => {
                        if socket.send(Message::Binary(frame)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    shared.clients.fetch_sub(1, Ordering::Relaxed);
}

fn browser_control(command: BrowserControl) -> Control {
    match command {
        BrowserControl::Pause => Control::Pause,
        BrowserControl::Resume => Control::Resume,
        BrowserControl::Reset => Control::Reset,
        BrowserControl::Step => Control::Step,
        BrowserControl::Preset { preset } => Control::Preset(preset),
        BrowserControl::View { pass } => Control::View(pass),
        BrowserControl::Resolution { width, height } => Control::Resolution(width, height),
        BrowserControl::TimeScale { value } => Control::TimeScale(value),
        BrowserControl::Uniform { name, value } => Control::Uniform { name, value },
        BrowserControl::Mouse {
            x,
            y,
            down,
            clicked,
        } => Control::Mouse {
            x,
            y,
            down,
            clicked,
        },
        BrowserControl::Key {
            code,
            down,
            pressed,
        } => Control::Key {
            code,
            down,
            pressed,
        },
    }
}

const INDEX_HTML: &str = crate::include_file!("preview/app.html");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_preset_control_supports_named_and_base_presets() {
        let named: BrowserControl =
            serde_json::from_str(r#"{"type":"preset","preset":"medium"}"#).unwrap();
        match browser_control(named) {
            Control::Preset(Some(name)) => assert_eq!(name, "medium"),
            other => panic!("unexpected control: {other:?}"),
        }

        let base: BrowserControl =
            serde_json::from_str(r#"{"type":"preset","preset":null}"#).unwrap();
        assert!(matches!(browser_control(base), Control::Preset(None)));
    }
}
