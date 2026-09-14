//! Compiled-in credential helpers: the `builtin:<mechanism>` auth-provider namespace.
//!
//! A provider preset can name a credential *mechanism* the way user config names an
//! `[auth_provider.<name>]` table, without shipping a credential or requiring the user to
//! write one. The mechanism runs in-process instead of spawning a helper command, but flows
//! through exactly the same mint/cache/rotate machinery ([`super::auth_provider`]): single
//! flight, TTL with refresh skew, the 401 fresh-mint guard, in-memory-only storage.
//!
//! A user `[auth_provider."builtin:..."]` table shadows the compiled mechanism (with a
//! reserved-namespace warning), so a broken builtin can be worked around locally.

use super::auth_provider::AuthProviderConfig;
use super::token_output::ParsedTokenOutput;

/// Auth-provider names in this namespace resolve against the compiled registry when no user
/// table defines them.
pub(crate) const BUILTIN_PROVIDER_PREFIX: &str = "builtin:";

/// GitHub Copilot token exchange: a durable GitHub token (env today, `grok login --provider
/// github-copilot` later) is exchanged for the short-lived Copilot API bearer.
const GITHUB_COPILOT_EXCHANGE: &str = "github-copilot-exchange";

/// Default exchange endpoint; overridable for tests via `GROK_COPILOT_TOKEN_EXCHANGE_URL`.
const GITHUB_COPILOT_EXCHANGE_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// Environment variables that may hold the durable GitHub token, in priority order.
const GITHUB_TOKEN_ENV_VARS: [&str; 3] = ["GITHUB_COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"];

/// The mechanism id for a `builtin:*` provider name, when the mechanism exists.
fn known_mechanism(provider_name: &str) -> Option<&str> {
    let mechanism = provider_name.strip_prefix(BUILTIN_PROVIDER_PREFIX)?;
    (mechanism == GITHUB_COPILOT_EXCHANGE).then_some(mechanism)
}

/// The trusted config for a `builtin:<mechanism>` provider name: `Some` only when the
/// mechanism is compiled in, so an unknown name still fails closed like any missing table.
pub(crate) fn builtin_provider_config(provider_name: &str) -> Option<AuthProviderConfig> {
    known_mechanism(provider_name).map(|mechanism| AuthProviderConfig {
        builtin: Some(mechanism.to_owned()),
        ..AuthProviderConfig::default()
    })
}

/// Run the compiled mechanism once and parse its token. Called under the provider slot lock by
/// [`super::auth_provider`]'s mint path, so the same single-flight and caching rules apply as
/// for a helper command.
pub(crate) async fn mint(mechanism: &str) -> anyhow::Result<ParsedTokenOutput> {
    match mechanism {
        GITHUB_COPILOT_EXCHANGE => mint_github_copilot().await,
        other => anyhow::bail!("unknown builtin credential helper '{other}'"),
    }
}

/// Exchange response shape (`GET /copilot_internal/v2/token`): the bearer plus a unix-seconds
/// expiry. Unknown fields are ignored so API drift doesn't break the parse.
#[derive(serde::Deserialize)]
struct CopilotTokenResponse {
    token: String,
    #[serde(default)]
    expires_at: Option<i64>,
}

async fn mint_github_copilot() -> anyhow::Result<ParsedTokenOutput> {
    let github_token = GITHUB_TOKEN_ENV_VARS
        .iter()
        .find_map(|var| std::env::var(var).ok().filter(|v| !v.trim().is_empty()))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no GitHub token found: set {} (a token with Copilot access)",
                GITHUB_TOKEN_ENV_VARS.join(", ")
            )
        })?;
    let url = std::env::var("GROK_COPILOT_TOKEN_EXCHANGE_URL")
        .ok
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| GITHUB_COPILOT_EXCHANGE_URL.to_owned());
    let response = reqwest::Client::new()
        .get(&url)
        .header(
            reqwest::header::AUTHORIZATION,
            format!("token {}", github_token.trim()),
        )
        // The GitHub API rejects requests without a User-Agent.
        .header(reqwest::header::USER_AGENT, "GrokBuild")
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("token exchange request failed: {e}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "token exchange returned {status}: {} (is the GitHub token valid and \
             Copilot-enabled?)",
            crate::util::truncate(body.trim(), 300)
        );
    }
    let parsed: CopilotTokenResponse = response
        .json()
        .await
        .map_err(|e| anyhow::anyhow!("token exchange returned a non-token payload: {e}"))?;
    let token = parsed.token.trim().to_owned();
    if token.is_empty() {
        anyhow::bail!("token exchange returned an empty token");
    }
    if token.contains(char::is_control) {
        anyhow::bail!("token exchange returned a token with control characters");
    }
    Ok(ParsedTokenOutput {
        access_token: token,
        refresh_token: None,
        expires_at: parsed
            .expires_at
            .and_then(|secs| chrono::DateTime::from_timestamp(secs, 0)),
        issuer: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_builtin_name_resolves_to_a_usable_config() {
        let config = builtin_provider_config("builtin:github-copilot-exchange")
            .expect("mechanism is compiled in");
        assert_eq!(config.builtin.as_deref(), Some("github-copilot-exchange"));
        assert!(config.is_usable(), "builtin config must be usable");
        assert!(config.command.is_empty(), "builtin never has a command");
    }

    #[test]
    fn unknown_builtin_name_fails_closed() {
        assert!(builtin_provider_config("builtin:nope").is_none());
        assert!(builtin_provider_config("github-copilot-exchange").is_none());
        assert!(builtin_provider_config("").is_none());
    }

    #[tokio::test]
    async fn unknown_mechanism_mint_errors() {
        let err = mint("nope").await.expect_err("unknown mechanism");
        assert!(err.to_string().contains("unknown builtin"));
    }
}
