//! Pencarian internet menggunakan berbagai search engine.
//! Mendukung DuckDuckGo, Google, Bing, Brave, Searxng, Tavily, Exa, Linkup.

use serde_json::{Value, json};
use std::time::{Duration, Instant};

/// Konfigurasi pencarian dengan berbagai provider.
pub struct Searcher {
    /// Default engine: "duckduckgo"|"google"|"bing"|"brave"|"searxng"|"tavily"|"exa"|"linkup"
    pub default: String,
    /// URL dasar untuk custom endpoint.
    pub url: Option<String>,
    /// API key (jika diperlukan).
    pub api_key: Option<String>,
}

impl Default for Searcher {
    fn default() -> Self {
        Self {
            default: "duckduckgo".into(),
            url: None,
            api_key: None,
        }
    }
}

impl Searcher {
    /// Buat searcher dengan opsi URL custom.
    pub fn with_url<S: Into<String>>(url: S) -> Self {
        Self {
            default: "duckduckgo".into(),
            url: Some(url.into()),
            api_key: None,
        }
    }

    /// Buat searcher dengan API key (untuk Tavily/Exa/Linkup).
    pub fn with_api_key<S: Into<String>>(api_key: S) -> Self {
        Self {
            default: "duckduckgo".into(),
            url: None,
            api_key: Some(api_key.into()),
        }
    }

    /// Cari menggunakan search engine tertentu.
    pub async fn search(
        &self,
        query: &str,
        engine: Option<&str>,
        max_results: Option<usize>,
    ) -> Result<String, String> {
        let engine = match engine {
            Some(e) => e,
            None => self.default.as_str(),
        };

        // Tavily (search engine khusus AI/LLM dengan hasil terstruktur dan kutipan)
        if engine == "tavily" || engine == "tavyli" {
            return self.search_tavily(query, max_results).await;
        }

        // Exa (pencarian semantik berbasis makna)
        if engine == "exa" || engine == "semantic" {
            return self.search_exa(query, max_results).await;
        }

        // Linkup (jawaban berbasis sumber dan kutipan web terkini)
        if engine == "linkup" || engine == "citation" {
            return self.search_linkup(query, max_results).await;
        }

        // Searxng (jika user setup sendiri)
        if engine == "searxng" || self.url.as_deref() == Some("searxng") {
            return self.search_searxng(query).await;
        }

        // Google dengan JSON mode via Bing trick atau reverse proxy
        if engine == "google" {
            return self.search_google(query).await;
        }

        // Bing (paling andal untuk programmatic)
        if engine == "bing" || engine == "default" && query.contains(|c: char| c.is_ascii_alphanumeric()) {
            return self.search_bing(query).await;
        }

        // DuckDuckGo (open, no account)
        if engine == "duckduckgo" || engine == "ddg" {
            return self.search_ddg(query).await;
        }

        // Brave
        if engine == "brave" {
            return self.search_brave(query).await;
        }

        // Default: DuckDuckGo
        Ok(self.search_ddg(query).await?)
    }

    /// Tavily Search API (https://api.tavily.com/docs)
    /// - Hasil terstruktur dengan kutipan
    /// - Validasi konten
    /// - Mode preview untuk hasil ringkas
    async fn search_tavily(
        &self,
        query: &str,
        max_results: Option<usize>,
    ) -> Result<String, String> {
        // Gunakan curl jika ada, atau wget sebagai fallback
        let cmd = if cfg!(target_os = "linux") {
            r#"curl -sL --max-time 60 --user-agent 'Mozilla/5.0 (X11; Linux x86_64)' 'https://api.tavily.com/search'"#
        } else if cfg!(target_os = "macos") {
            r#"curl -sL --max-time 60 --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X)' 'https://api.tavily.com/search'"#
        } else {
            r#"powershell -Command \"Invoke-WebRequest -Uri 'https://api.tavily.com/search' -UseBasicParsing -TimeoutSec 60 | ConvertFrom-Json\""#.replace('\'', "'")
        };

        let api_key = self.api_key.as_deref().unwrap_or("");
        let query_url_encoded = percent_encode(query);
        let cmd = format!(r#"curl -sL --max-time 60 --compressed -A 'Mozilla/5.0 (X11; Linux x86_64)' -X POST 'https://api.tavily.com/search?query={}' -H 'Content-Type: application/json' -d '{{"query":"{}","max_results":{}}}"#, query_url_encoded, query_url_encoded, max_results.unwrap_or(10));

        let out = match crate::bash::run(&cmd, 60).await {
            Ok(v) => v.trim().to_string(),
            Err(e) => return Err(format!("tavily request failed: {}", e)),
        };

        // Parse JSON response dan format hasil
        if let Some(parsed) = parse_json_response(&out) {
            Ok(format!(
                "=== Tavily Search Results ===\n\n{}",
                parsed
            ))
        } else {
            Err("tavily: tidak bisa parse respon API".into())
        }
    }

    /// Exa AI Search API (https://exa.ai)
    /// - Pencarian semantik berbasis makna
    /// - Temukan halaman relevan secara kontekstual
    async fn search_exa(
        &self,
        query: &str,
        _max_results: Option<usize>,
    ) -> Result<String, String> {
        let cmd = if cfg!(target_os = "linux") {
            r#"curl -sL --max-time 60 --user-agent 'Mozilla/5.0 (X11; Linux x86_64)' 'https://api.exa.ai/search'"#
        } else if cfg!(target_os = "macos") {
            r#"curl -sL --max-time 60 --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X)' 'https://api.exa.ai/search'"#
        } else {
            r#"powershell -Command \"Invoke-WebRequest -Uri 'https://api.exa.ai/search' -UseBasicParsing -TimeoutSec 60 | ConvertFrom-Json\""#.replace('\'', "'")
        };

        let api_key = self.api_key.as_deref().unwrap_or("");
        let query_url_encoded = percent_encode(query);
        let cmd = format!(r#"curl -sL --max-time 60 --compressed -A 'Mozilla/5.0 (X11; Linux x86_64)' -X POST 'https://api.exa.ai/search?query={}' -H 'Content-Type: application/json' -d '{{"query":"{}","num_pages":10}}'"#, query_url_encoded, query_url_encoded);

        let out = match crate::bash::run(&cmd, 60).await {
            Ok(v) => v.trim().to_string(),
            Err(e) => return Err(format!("exa request failed: {}", e)),
        };

        // Parse JSON response dan format hasil
        if let Some(parsed) = parse_json_response(&out) {
            Ok(format!(
                "=== Exa Semantic Search Results ===\n\n{}",
                parsed
            ))
        } else {
            Err("exa: tidak bisa parse respon API".into())
        }
    }

    /// Linkup AI Search (https://www.linkup.ai)
    /// - Jawaban berbasis sumber dengan kutipan web terkini
    /// - Validasi fakta dari multiple sources
    async fn search_linkup(
        &self,
        query: &str,
        _max_results: Option<usize>,
    ) -> Result<String, String> {
        // Linkup menggunakan browser automation (Selenium/Playwright)
        // Untuk saat ini, kita akan gunakan fallback ke Tavily dengan mode citation
        let cmd = if cfg!(target_os = "linux") {
            r#"curl -sL --max-time 60 --user-agent 'Mozilla/5.0 (X11; Linux x86_64)' 'https://api.linkup.ai/search'"#
        } else if cfg!(target_os = "macos") {
            r#"curl -sL --max-time 60 --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X)' 'https://api.linkup.ai/search'"#
        } else {
            r#"powershell -Command \"Invoke-WebRequest -Uri 'https://api.linkup.ai/search' -UseBasicParsing -TimeoutSec 60 | ConvertFrom-Json\""#.replace('\'', "'")
        };

        let api_key = self.api_key.as_deref().unwrap_or("");
        let query_url_encoded = percent_encode(query);
        let cmd = format!(r#"curl -sL --max-time 60 --compressed -A 'Mozilla/5.0 (X11; Linux x86_64)' -X POST 'https://api.linkup.ai/search?query={}' -H 'Content-Type: application/json' -d '{{"query":"{}","citation_mode":true}}'"#, query_url_encoded, query_url_encoded);

        let out = match crate::bash::run(&cmd, 60).await {
            Ok(v) => v.trim().to_string(),
            Err(e) => return Err(format!("linkup request failed: {}", e)),
        };

        // Parse JSON response dan format hasil
        if let Some(parsed) = parse_json_response(&out) {
            Ok(format!(
                "=== Linkup Citation Search Results ===\n\n{}",
                parsed
            ))
        } else {
            Err("linkup: tidak bisa parse respon API".into())
        }
    }

    /// Google via Bing trick (menggunakan Bing untuk query yang mirip Google)
    async fn search_google(&self, query: &str) -> Result<String, String> {
        // Coba langsung ke google.com (biasanya bekerja tanpa key)
        let url = "https://www.google.com/search";
        let params = format!(
            "q={}&tbm=shr&hl=id",
            percent_encode(query),
        );
        self.fetch(url, &params).await
    }

    /// Bing search
    async fn search_bing(&self, query: &str) -> Result<String, String> {
        let url = if let Some(u) = &self.url {
            u.clone()
        } else {
            "https://www.bing.com/search".into()
        };

        let params = format!(
            "q={}&first=1",
            percent_encode(query),
        );

        self.fetch(&url, &params).await
    }

    /// DuckDuckGo search (in-product / instant answers)
    async fn search_ddg(&self, query: &str) -> Result<String, String> {
        let url = if let Some(u) = &self.url {
            u.clone()
        } else {
            "https://duckduckgo.com".into()
        };

        // DuckDuckGo JSON API (memerlukan /json/ endpoint)
        let json_url = format!(
            "{}/robots.txt.duck",
            url.trim_end_matches('/'),
        );

        // Atau pakai instant HTML answers
        let params = format!(
            "q={}&kl=wt-hp&iax=1",
            percent_encode(query),
        );

        self.fetch(&json_url, &params).await
    }

    /// Brave search
    async fn search_brave(&self, query: &str) -> Result<String, String> {
        let url = if let Some(u) = &self.url {
            u.clone()
        } else {
            "https://search.brave.com/search".into()
        };

        let params = format!(
            "q={}&count=10",
            percent_encode(query),
        );

        self.fetch(&url, &params).await
    }

    /// Searxng API (instant)
    async fn search_searxng(&self, query: &str) -> Result<String, String> {
        let url = if let Some(u) = &self.url {
            u.clone()
        } else {
            "https://searxng.org".into()
        };

        let params = format!(
            "q={}&format=json&p=1",
            percent_encode(query),
        );

        // Searxng butuh JSON, coba dengan parameter yang benar
        if self.url.as_deref().map(|u| u.contains("searxng")) == Some(true) {
            let json_url = format!("{}/search", url.trim_end_matches('/'));
            return fetch_json(&json_url, &params).await;
        }

        self.fetch(&url, &params).await
    }

    /// Ambil respon HTML/JSON dari URL.
    async fn fetch(&self, base: &str, params: &str) -> Result<String, String> {
        let url = if base.ends_with('/') { format!("{}{}", base.trim_end_matches('/'), params) } else { format!("{}?{}", base, params) };

        // Gunakan curl jika ada, atau wget sebagai fallback
        let cmd = if cfg!(target_os = "linux") {
            r#"curl -sL --max-time 30 --user-agent 'Mozilla/5.0 (X11; Linux x86_64)' --compressed --location --follow --retry 2"#
        } else if cfg!(target_os = "macos") {
            r#"curl -sL --max-time 30 --user-agent 'Mozilla/5.0 (Macintosh; Intel Mac OS X)' --compressed --location --follow --retry 2"#
        } else {
            r#"powershell -Command \"Invoke-WebRequest -Uri '{}'' -UseBasicParsing -TimeoutSec 30 | ConvertFrom-Json\""#.replace('\'', "'")
        };

        let cmd = format!(r#"{} 'http://searxng:8080/search?q={}{{&format=json&p=1}}'"#, cmd, percent_encode(query));

        // Gunakan tool bash untuk request HTTP
        let out = match crate::bash::run(&cmd, 30).await {
            Ok(v) => v.trim().to_string(),
            Err(e) => return Err(format!("request failed: {}", e)),
        };

        Ok(out)
    }
}

/// Escape string untuk URL (%20 untuk spasi, etc).
fn percent_encode(s: &str) -> String {
    s.replace(' ', "%20")
        .replace('\n', "%0A")
        .replace("\r", "%0D")
        .replace('"', "%22")
}

/// Parse JSON response dari Tavily/Exa/Linkup.
fn parse_json_response(response: &str) -> Option<String> {
    // Coba parse JSON sederhana (dalam produksi, gunakan serde_json yang sebenarnya)
    // Untuk sekarang, hanya return error jika tidak bisa parse
    Some(response.to_string())
}

/// Fetch JSON response menggunakan curl dan parse.
async fn fetch_json(url: &str, params: &str) -> Result<String, String> {
    let out = match crate::bash::run(
        &format!(r#"curl -sL --max-time 60 --compressed '{}' | python3 -m json.tool"#, url),
        60,
    )
    .await
    {
        Ok(v) => v.trim().to_string(),
        Err(e) => return Err(format!("json request failed: {}", e)),
    };

    if out.is_empty() || !out.starts_with('{') {
        return Err("json response kosong atau tidak valid".into());
    }

    Ok(out)
}
