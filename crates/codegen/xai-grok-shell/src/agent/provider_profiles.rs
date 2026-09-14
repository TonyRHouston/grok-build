//! Built-in provider presets: the descriptor that answers "how do I talk to this vendor".
//!
//! A provider used to be a bag of loose TOML fields the user assembled by hand — there was no
//! type that owned provider identity, and no built-in knowledge of OpenAI, Anthropic, or Copilot.
//! [`ProviderProfile`] is that type, and [`DEFAULT_PROVIDERS_JSON`] is the data behind it.
//!
//! The presets are data, not code, mirroring `xai-grok-models/default_models.json`: adding a
//! provider is a JSON edit, and the file can later be served by remote settings the way the model
//! catalog already is.
//!
//! Layering, lowest priority first:
//!
//! 1. the built-in [`ProviderProfile`] for the id,
//! 2. the user's `[model_providers.<id>]` block ([`ModelProviderConfig`]),
//! 3. the model's own `[model.<id>]` keys.
//!
//! A preset never carries a credential. It names the environment variable(s) that hold one, so
//! reaching a third-party provider requires no secret in `config.toml`.

use std::sync::LazyLock;

use indexmap::IndexMap;

use super::model_providers::ModelProviderConfig;
use crate::agent::config::EnvKeys;
use crate::sampling::ApiBackend;
use xai_grok_sampler::AuthScheme;

/// The raw preset JSON, embedded at compile time.
pub const DEFAULT_PROVIDERS_JSON: &str = include_str!("default_providers.json");

/// Connection defaults for one vendor.
///
/// Every field except `id`/`name` is optional: a profile only states what it actually knows, so a
/// user override and a preset merge without either having to spell out the other's fields.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct ProviderProfile {
    pub id: String,
    /// Human-readable label, for diagnostics and the model picker.
    pub name: String,
    /// Inference base URL, e.g. `https://api.anthropic.com/v1`.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Wire dialect this vendor speaks.
    #[serde(default)]
    pub api_backend: Option<ApiBackend>,
    /// How the credential is presented: `Authorization: Bearer` or `x-api-key`.
    #[serde(default)]
    pub auth_scheme: Option<AuthScheme>,
    /// Environment variable(s) holding the API key, in priority order.
    #[serde(default)]
    pub env_key: Option<EnvKeys>,
    /// Headers every request to this vendor needs (e.g. `anthropic-version`), so they are never
    /// user-authored.
    #[serde(default)]
    pub extra_headers: IndexMap<String, String>,
    /// Query parameters folded into every request URL.
    #[serde(default)]
    pub query_params: IndexMap<String, String>,
    /// Header name to environment variable, resolved at client build and never persisted.
    #[serde(default)]
    pub env_http_headers: IndexMap<String, String>,
    /// Capability default: total context window in tokens.
    #[serde(default)]
    pub context_window: Option<u64>,
    /// Capability default: whether to ask the endpoint for per-chunk tool-call deltas.
    /// Some vendors reject the field, which is exactly why it belongs on the profile.
    #[serde(default)]
    pub stream_tool_calls: Option<bool>,
    /// Capability default: whether models from this vendor accept a reasoning-effort setting.
    #[serde(default)]
    pub supports_reasoning_effort: Option<bool>,
    /// Compiled-in credential helper (`builtin:<mechanism>`) models inherit when the user
    /// configured no credential of their own; the preset names a *mechanism*, never a secret.
    /// E.g. the GitHub Copilot preset exchanges a durable GitHub token for the short-lived
    /// Copilot bearer.
    #[serde(default)]
    pub auth_helper: Option<String>,
    /// True only for the xAI-operated provider.
    #[serde(default)]
    pub first_party: bool,
}

#[derive(serde::Deserialize)]
struct DefaultProviders {
    providers: Vec<ProviderProfile>,
}

static PROFILES: LazyLock<IndexMap<String, ProviderProfile>> = LazyLock::new(|| {
    let parsed: DefaultProviders = serde_json::from_str(DEFAULT_PROVIDERS_JSON)
        .expect("default_providers.json: invalid JSON or missing a required field");
    let mut map = IndexMap::with_capacity(parsed.providers.len());
    for profile in parsed.providers {
        // Baked-in JSON: a duplicate id is a developer error, not a user one.
        assert!(
            !map.contains_key(&profile.id),
            "default_providers.json: duplicate provider id '{}'",
            profile.id,
        );
        map.insert(profile.id.clone(), profile);
    }
    map
});

/// The built-in profile for `id`, if one exists.
pub fn builtin_profile(id: &str) -> Option<&'static ProviderProfile> {
    PROFILES.get(id)
}

/// Every built-in profile, in declaration order.
pub fn builtin_profiles() -> impl Iterator<Item = &'static ProviderProfile> {
    PROFILES.values()
}

impl ProviderProfile {
    /// This profile as a [`ModelProviderConfig`], so preset-only providers flow through exactly
    /// the same merge path as user-declared ones.
    pub(crate) fn to_provider_config(&self) -> ModelProviderConfig {
        ModelProviderConfig {
            base_url: self.base_url.clone(),
            api_base_url: None,
            env_key: self.env_key.clone(),
            api_key: None,
            api_backend: self.api_backend.clone(),
            auth_scheme: self.auth_scheme,
            extra_headers: self.extra_headers.clone(),
            query_params: self.query_params.clone(),
            env_http_headers: self.env_http_headers.clone(),
            // The helper is a fallback behind env_key: `resolve_credentials` tries the model's
            // own credential (env/api_key) before the provider mint, so an env var always wins.
            auth_provider: self.auth_helper.clone(),
            auth: None,
            context_window: self.context_window,
            stream_tool_calls: self.stream_tool_calls,
            supports_reasoning_effort: self.supports_reasoning_effort,
        }
    }
}

/// Resolve `id` to the provider config a model should inherit from.
///
/// The user's block wins field by field; the built-in preset fills only what the user left unset.
/// Returns `None` when neither source knows the id, which is what makes the "undefined provider"
/// warning (and its fail-closed credential handling) still fire for a typo.
pub(crate) fn resolve_provider(
    user_providers: &IndexMap<String, ModelProviderConfig>,
    id: &str,
) -> Option<ModelProviderConfig> {
    match (user_providers.get(id), builtin_profile(id)) {
        (Some(user), Some(profile)) => Some(user.with_profile_defaults(profile)),
        (Some(user), None) => Some(user.clone()),
        (None, Some(profile)) => Some(profile.to_provider_config()),
        (None, None) => None,
    }
}

/// True when `id` names a provider from either source. Used to decide whether a
/// `model_provider = "<id>"` reference is a typo.
pub(crate) fn provider_is_known(
    user_providers: &IndexMap<String, ModelProviderConfig>,
    declared: &std::collections::HashSet<&str>,
    id: &str,
) -> bool {
    user_providers.contains_key(id) || declared.contains(id) || builtin_profile(id).is_some()
}

/// The registrable domain (last two labels) of a URL's host, lowercased.
fn registrable_domain(host: &str) -> Option<String> {
    let mut labels = host.rsplit('.');
    let tld = labels.next().filter(|l| !l.is_empty())?;
    let second = labels.next().filter(|l| !l.is_empty())?;
    Some(format!("{second}.{tld}").to_ascii_lowercase())
}

/// True when `url` routes to a host owned by a known third-party vendor — the registrable domain
/// of any non-first-party preset's base URL (e.g. `openai.com`, `anthropic.com`,
/// `githubcopilot.com`), including subdomains such as regional endpoints.
///
/// This is the structural deny behind session isolation: `resolve_credentials` refuses to attach a
/// first-party credential (session bearer or `XAI_API_KEY`) to these hosts *under any
/// configuration*, even for a hand-written `[model.<id>]` with no `model_provider` tag. Presets
/// are data, so the deny set updates with `default_providers.json`. Unparseable URLs are not
/// vendor hosts; the scheme is ignored so the deny also covers plain-HTTP variants.
pub(crate) fn is_third_party_vendor_url(url: &str) -> bool {
    let Some(host) = reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_ascii_lowercase))
    else {
        return false;
    };
    builtin_profiles()
        .filter(|p| !p.first_party)
        .filter_map(|p| p.base_url.as_deref())
        .filter_map(|base| {
            reqwest::Url::parse(base)
                .ok()
                .and_then(|u| u.host_str().and_then(registrable_domain))
        })
        .any(|vendor| host == vendor || host.ends_with(&format!(".{vendor}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse_and_cover_the_documented_vendors() {
        for id in ["xai", "openai", "codex", "anthropic", "github-copilot"] {
            assert!(
                builtin_profile(id).is_some(),
                "missing built-in provider preset '{id}'"
            );
        }
        assert_eq!(builtin_profiles().count(), 5);
    }

    #[test]
    fn only_xai_is_first_party() {
        let first_party: Vec<&str> = builtin_profiles()
            .filter(|p| p.first_party)
            .map(|p| p.id.as_str())
            .collect();
        assert_eq!(first_party, vec!["xai"]);
    }

    /// A preset that shipped a credential would put a secret in the binary and defeat the whole
    /// point of `env_key`.
    #[test]
    fn no_preset_carries_a_credential() {
        for profile in builtin_profiles() {
            assert!(
                profile.env_key.as_ref().is_some_and(|k| !k.is_empty()),
                "provider '{}' must name an env var to read its key from",
                profile.id
            );
            let config = profile.to_provider_config();
            assert!(
                config.api_key.is_none(),
                "provider '{}' must not ship a static api_key",
                profile.id
            );
            for (header, value) in &profile.extra_headers {
                assert!(
                    !header.eq_ignore_ascii_case("authorization")
                        && !header.eq_ignore_ascii_case("x-api-key"),
                    "provider '{}' puts a credential header ({header}={value}) in a preset; \
                     use auth_scheme + env_key instead",
                    profile.id
                );
            }
        }
    }

    #[test]
    fn anthropic_preset_speaks_messages_with_x_api_key_and_a_version_header() {
        let anthropic = builtin_profile("anthropic").expect("anthropic preset");
        assert_eq!(anthropic.api_backend, Some(ApiBackend::Messages));
        assert_eq!(anthropic.auth_scheme, Some(AuthScheme::XApiKey));
        assert_eq!(
            anthropic
                .extra_headers
                .get("anthropic-version")
                .map(String::as_str),
            Some("2023-06-01"),
            "users must never have to hand-write anthropic-version"
        );
        assert_eq!(
            anthropic.env_key.as_ref().and_then(EnvKeys::primary),
            Some("ANTHROPIC_API_KEY")
        );
    }

    #[test]
    fn codex_preset_uses_the_responses_backend() {
        let codex = builtin_profile("codex").expect("codex preset");
        assert_eq!(codex.api_backend, Some(ApiBackend::Responses));
        assert_eq!(codex.auth_scheme, Some(AuthScheme::Bearer));
        let openai = builtin_profile("openai").expect("openai preset");
        assert_eq!(openai.api_backend, Some(ApiBackend::ChatCompletions));
    }

    #[test]
    fn builtin_preset_resolves_without_any_user_block() {
        let empty = IndexMap::new();
        let resolved = resolve_provider(&empty, "anthropic").expect("preset should resolve alone");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://api.anthropic.com/v1")
        );
        assert!(resolve_provider(&empty, "not-a-provider").is_none());
    }

    #[test]
    fn user_block_overrides_the_preset_field_by_field() {
        let mut user = IndexMap::new();
        user.insert(
            "anthropic".to_owned(),
            ModelProviderConfig {
                base_url: Some("https://gateway.example/anthropic".to_owned()),
                ..Default::default()
            },
        );
        let resolved = resolve_provider(&user, "anthropic").expect("merged provider");
        assert_eq!(
            resolved.base_url.as_deref(),
            Some("https://gateway.example/anthropic"),
            "the user's base_url must win"
        );
        assert_eq!(
            resolved.api_backend,
            Some(ApiBackend::Messages),
            "fields the user left unset still come from the preset"
        );
        assert_eq!(
            resolved
                .extra_headers
                .get("anthropic-version")
                .map(String::as_str),
            Some("2023-06-01")
        );
    }
}
