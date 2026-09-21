use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

const BLIND_FORMAT: u32 = 1;
const SEALED_MAGIC: &[u8; 8] = b"STBLND1\0";

#[derive(Debug)]
pub(super) struct BlindPlan {
    session_id: String,
    seed: [u8; 32],
    order: Vec<usize>,
    labels: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BlindSession {
    format: u32,
    session_id: String,
    project: String,
    frame: i32,
    pass: String,
    width: u32,
    height: u32,
    contact_sheet: PathBuf,
    variants: Vec<BlindPublicVariant>,
    sealed_mapping: PathBuf,
    judgment: PathBuf,
    reveal: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlindPublicVariant {
    label: String,
    output: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct BlindMapping {
    format: u32,
    session_id: String,
    variants: Vec<BlindMappingVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlindMappingVariant {
    label: String,
    original_index: usize,
    output: PathBuf,
    set: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct BlindJudgment {
    format: u32,
    session_id: String,
    selected: String,
    reasoning: String,
    recorded_at_unix_ms: u128,
}

impl BlindPlan {
    pub(super) fn new(variant_count: usize, entropy_context: &str) -> Result<Self> {
        if variant_count < 2 {
            bail!("blind comparison requires at least two sweep variants");
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock is before UNIX epoch")?
            .as_nanos();
        let mut hasher = Sha256::new();
        hasher.update(b"shadertoy-blind-v1");
        hasher.update(entropy_context.as_bytes());
        hasher.update(now.to_le_bytes());
        hasher.update(std::process::id().to_le_bytes());
        let digest = hasher.finalize();
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&digest);

        let mut order = (0..variant_count).collect::<Vec<_>>();
        let mut state = u64::from_le_bytes(seed[..8].try_into().expect("eight-byte seed slice"));
        if state == 0 {
            state = 0x9e37_79b9_7f4a_7c15;
        }
        for index in (1..order.len()).rev() {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let random = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
            let swap_with = (random as usize) % (index + 1);
            order.swap(index, swap_with);
        }

        let labels = (0..variant_count).map(blind_label).collect::<Vec<_>>();
        let session_id = seed[..8]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(Self {
            session_id,
            seed,
            order,
            labels,
        })
    }

    pub(super) fn order(&self) -> &[usize] {
        &self.order
    }

    pub(super) fn label(&self, index: usize) -> &str {
        &self.labels[index]
    }
}

pub(super) struct BlindSessionSpec<'a> {
    pub output_dir: &'a Path,
    pub project: &'a str,
    pub frame: i32,
    pub pass: &'a str,
    pub width: u32,
    pub height: u32,
    pub contact_sheet: &'a Path,
    pub outputs: &'a [PathBuf],
    pub variants: &'a [Vec<String>],
}

pub(super) fn write_blind_session(
    plan: &BlindPlan,
    spec: &BlindSessionSpec<'_>,
) -> Result<PathBuf> {
    if spec.outputs.len() != plan.order.len() || spec.variants.len() != plan.order.len() {
        bail!("blind session variant metadata does not match the rendered sweep");
    }

    let session_path = spec.output_dir.join("blind-session.json");
    let sealed_mapping_name = PathBuf::from(".blind-mapping.bin");
    let judgment_name = PathBuf::from("blind-judgment.json");
    let reveal_name = PathBuf::from("blind-reveal.json");
    let sealed_mapping = spec.output_dir.join(&sealed_mapping_name);
    let judgment = spec.output_dir.join(&judgment_name);
    let reveal = spec.output_dir.join(&reveal_name);

    // A fresh blind sweep starts a fresh decision lifecycle even when reusing an output directory.
    for stale in [&judgment, &reveal] {
        if stale.exists() {
            fs::remove_file(stale).with_context(|| {
                format!("failed to remove stale blind artifact {}", stale.display())
            })?;
        }
    }

    let public_variants = spec
        .outputs
        .iter()
        .enumerate()
        .map(|(position, output)| BlindPublicVariant {
            label: plan.label(position).to_string(),
            output: output.clone(),
        })
        .collect::<Vec<_>>();

    let mapping_variants = plan
        .order
        .iter()
        .enumerate()
        .map(|(position, original_index)| BlindMappingVariant {
            label: plan.label(position).to_string(),
            original_index: *original_index,
            output: spec.outputs[position].clone(),
            set: spec.variants[*original_index].clone(),
        })
        .collect::<Vec<_>>();

    let session = BlindSession {
        format: BLIND_FORMAT,
        session_id: plan.session_id.clone(),
        project: spec.project.to_string(),
        frame: spec.frame,
        pass: spec.pass.to_string(),
        width: spec.width,
        height: spec.height,
        contact_sheet: spec.contact_sheet.to_path_buf(),
        variants: public_variants,
        sealed_mapping: sealed_mapping_name,
        judgment: judgment_name,
        reveal: reveal_name,
    };
    let mapping = BlindMapping {
        format: BLIND_FORMAT,
        session_id: plan.session_id.clone(),
        variants: mapping_variants,
    };

    write_pretty_json(&session_path, &session)?;
    write_sealed_mapping(&sealed_mapping, &plan.seed, &mapping)?;
    Ok(session_path)
}

pub fn judge_blind(options: &BlindJudgeOptions) -> Result<Output> {
    let session_path = resolve_session_path(&options.session);
    let session: BlindSession = read_json(&session_path, "blind session")?;
    validate_session(&session)?;
    let session_dir = session_parent(&session_path)?;
    let reveal_path = resolve_artifact_path(session_dir, &session.reveal);
    let judgment_path = resolve_artifact_path(session_dir, &session.judgment);

    if reveal_path.exists() {
        bail!(
            "blind session '{}' has already been revealed; judgment is closed",
            session.session_id
        );
    }
    if judgment_path.exists() {
        bail!(
            "blind session '{}' already has a recorded judgment at {}",
            session.session_id,
            judgment_path.display()
        );
    }

    let selected = options.pick.trim().to_ascii_uppercase();
    if !session
        .variants
        .iter()
        .any(|variant| variant.label == selected)
    {
        let valid = session
            .variants
            .iter()
            .map(|variant| variant.label.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        bail!("unknown blind label '{selected}'; expected one of {valid}");
    }

    let reasoning = match (&options.reason, &options.reason_file) {
        (Some(reason), None) => reason.clone(),
        (None, Some(path)) => fs::read_to_string(path)
            .with_context(|| format!("failed to read reasoning file {}", path.display()))?,
        _ => bail!("provide exactly one of --reason or --reason-file"),
    };
    let reasoning = reasoning.trim().to_string();
    if reasoning.is_empty() {
        bail!("blind judgment reasoning must not be empty");
    }

    let judgment = BlindJudgment {
        format: BLIND_FORMAT,
        session_id: session.session_id.clone(),
        selected: selected.clone(),
        reasoning: reasoning.clone(),
        recorded_at_unix_ms: unix_time_ms()?,
    };
    write_pretty_json(&judgment_path, &judgment)?;

    Ok(Output {
        human: format!(
            "Recorded blind judgment for session {}: selected {}. Mapping remains sealed; run 'shadertoy blind reveal {}'.",
            session.session_id,
            selected,
            session_path.display()
        ),
        json: json!({
            "ok": true,
            "session_id": session.session_id,
            "session": session_path,
            "judgment": judgment_path,
            "selected": selected,
            "reasoning": reasoning,
            "revealed": false,
        }),
    })
}

pub fn reveal_blind(options: &BlindRevealOptions) -> Result<Output> {
    let session_path = resolve_session_path(&options.session);
    let session: BlindSession = read_json(&session_path, "blind session")?;
    validate_session(&session)?;
    let session_dir = session_parent(&session_path)?;
    let judgment_path = resolve_artifact_path(session_dir, &session.judgment);
    let mapping_path = resolve_artifact_path(session_dir, &session.sealed_mapping);
    let reveal_path = resolve_artifact_path(session_dir, &session.reveal);

    if !judgment_path.exists() {
        bail!(
            "blind session '{}' has no judgment yet; record one with 'shadertoy blind judge {} --pick LABEL --reason ...' before revealing",
            session.session_id,
            session_path.display()
        );
    }

    let judgment: BlindJudgment = read_json(&judgment_path, "blind judgment")?;
    if judgment.session_id != session.session_id {
        bail!("blind judgment belongs to a different session");
    }
    let mapping = read_sealed_mapping(&mapping_path)?;
    if mapping.session_id != session.session_id {
        bail!("sealed blind mapping belongs to a different session");
    }
    let selected_variant = mapping
        .variants
        .iter()
        .find(|variant| variant.label == judgment.selected)
        .cloned()
        .with_context(|| {
            format!(
                "judged label '{}' is missing from the sealed mapping",
                judgment.selected
            )
        })?;

    let report = json!({
        "format": BLIND_FORMAT,
        "session_id": session.session_id,
        "project": session.project,
        "frame": session.frame,
        "pass": session.pass,
        "width": session.width,
        "height": session.height,
        "contact_sheet": session.contact_sheet,
        "judgment": judgment,
        "selected_variant": selected_variant,
        "mapping": mapping.variants,
    });
    write_pretty_json_value(&reveal_path, &report)?;

    let mapping_text = report["mapping"]
        .as_array()
        .expect("mapping JSON is an array")
        .iter()
        .map(|variant| {
            let label = variant["label"].as_str().unwrap_or("?");
            let set = variant["set"]
                .as_array()
                .map(|values| {
                    values
                        .iter()
                        .filter_map(|value| value.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            format!("{label} = {set}")
        })
        .collect::<Vec<_>>()
        .join("; ");

    Ok(Output {
        human: format!(
            "Revealed blind session {} after judgment {}. {}. Combined report: {}",
            report["session_id"].as_str().unwrap_or("?"),
            report["judgment"]["selected"].as_str().unwrap_or("?"),
            mapping_text,
            reveal_path.display()
        ),
        json: json!({
            "ok": true,
            "session": session_path,
            "report": reveal_path,
            "result": report,
        }),
    })
}

fn resolve_session_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("blind-session.json")
    } else {
        path.to_path_buf()
    }
}

fn session_parent(session_path: &Path) -> Result<&Path> {
    session_path
        .parent()
        .context("blind session path has no parent directory")
}

fn resolve_artifact_path(session_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        session_dir.join(path)
    }
}

fn validate_session(session: &BlindSession) -> Result<()> {
    if session.format != BLIND_FORMAT {
        bail!(
            "unsupported blind session format {}; expected {}",
            session.format,
            BLIND_FORMAT
        );
    }
    if session.variants.len() < 2 {
        bail!("blind session must contain at least two variants");
    }
    Ok(())
}

fn write_sealed_mapping(path: &Path, seed: &[u8; 32], mapping: &BlindMapping) -> Result<()> {
    let payload = serde_json::to_vec(mapping).context("failed to serialize blind mapping")?;
    let encrypted = xor_stream(&payload, seed);
    let mut bytes = Vec::with_capacity(SEALED_MAGIC.len() + seed.len() + encrypted.len());
    bytes.extend_from_slice(SEALED_MAGIC);
    bytes.extend_from_slice(seed);
    bytes.extend_from_slice(&encrypted);
    fs::write(path, bytes)
        .with_context(|| format!("failed to write sealed blind mapping {}", path.display()))
}

fn read_sealed_mapping(path: &Path) -> Result<BlindMapping> {
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read sealed blind mapping {}", path.display()))?;
    let header_len = SEALED_MAGIC.len() + 32;
    if bytes.len() < header_len || &bytes[..SEALED_MAGIC.len()] != SEALED_MAGIC {
        bail!("invalid sealed blind mapping {}", path.display());
    }
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&bytes[SEALED_MAGIC.len()..header_len]);
    let payload = xor_stream(&bytes[header_len..], &seed);
    serde_json::from_slice(&payload)
        .with_context(|| format!("failed to decode sealed blind mapping {}", path.display()))
}

fn xor_stream(input: &[u8], seed: &[u8; 32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    for (block_index, chunk) in input.chunks(32).enumerate() {
        let mut hasher = Sha256::new();
        hasher.update(b"shadertoy-blind-seal-v1");
        hasher.update(seed);
        hasher.update((block_index as u64).to_le_bytes());
        let key = hasher.finalize();
        output.extend(chunk.iter().zip(key.iter()).map(|(byte, mask)| byte ^ mask));
    }
    output
}

fn blind_label(index: usize) -> String {
    let mut value = index + 1;
    let mut chars = Vec::new();
    while value > 0 {
        let digit = (value - 1) % 26;
        chars.push((b'A' + digit as u8) as char);
        value = (value - 1) / 26;
    }
    chars.iter().rev().collect()
}

fn unix_time_ms() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before UNIX epoch")?
        .as_millis())
}

fn write_pretty_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("failed to serialize JSON artifact")?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn write_pretty_json_value(path: &Path, value: &Value) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value).context("failed to serialize JSON artifact")?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, kind: &str) -> Result<T> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read {kind} {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {kind} {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blind_labels_use_spreadsheet_style_names() {
        assert_eq!(blind_label(0), "A");
        assert_eq!(blind_label(25), "Z");
        assert_eq!(blind_label(26), "AA");
        assert_eq!(blind_label(51), "AZ");
        assert_eq!(blind_label(52), "BA");
    }

    #[test]
    fn sealed_mapping_round_trips_without_plaintext_assignments() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(".blind-mapping.bin");
        let mapping = BlindMapping {
            format: BLIND_FORMAT,
            session_id: "session".into(),
            variants: vec![BlindMappingVariant {
                label: "A".into(),
                original_index: 0,
                output: PathBuf::from("A.png"),
                set: vec!["gain=0.8".into()],
            }],
        };
        let seed = [7u8; 32];
        write_sealed_mapping(&path, &seed, &mapping).unwrap();
        let raw = fs::read(&path).unwrap();
        assert!(
            !raw.windows(b"gain=0.8".len())
                .any(|window| window == b"gain=0.8")
        );
        let decoded = read_sealed_mapping(&path).unwrap();
        assert_eq!(decoded.variants[0].set, ["gain=0.8"]);
    }
}
