# aidaemon-secrets

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Where an agent keeps its secrets (API keys, bot tokens, OAuth tokens). `SecretStore` gets, sets and deletes a secret by key. `KeychainSecretStore` keeps them in the OS credential store under one service name, and switches to an env file when an environment variable you name turns the keychain off; `EnvSecretStore` is that env-file mode on its own (an `.env` file, falling back to process variables, keys upper-cased); `MemorySecretStore` keeps them in memory for tests. `names` builds the account names aidaemon's configuration uses.

```rust,ignore
use aidaemon_secrets::{KeychainSecretStore, KeychainSettings, SecretStore};

let store = KeychainSecretStore::new(KeychainSettings::new("my-agent"));
store.set("api_key", "sk-...")?;
let key = store.get("api_key")?;
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.20.0). MIT licensed. Docs: https://docs.rs/aidaemon-secrets
