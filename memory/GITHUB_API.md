# Peringatan GitHub API

Agen **tidak boleh** crawl HTML dari GitHub bila sudah memiliki URL repositori dari GitHub API atau Exa.

## Alasan:
- Meng-crawl HTML GitHub boros token
- Tidak reliable untuk parsing
- Output JSON API sudah cukup informatif

## Prinsip Penggunaan GitHub API:
1. Gunakan output JSON dari API resmi (search, releases, contributors, contents, pulls, issues, stargazers)
2. Jangan fetch HTML kecuali memang diperlukan (misalnya untuk preview README dengan API `GET .../readme`)
3. Semua tools GitHub hanya mengembalikan JSON dari API, bukan hasil crawl HTML

## Tools GitHub yang tersedia:
- Search API (`search/repositories`, `search/code`)
- Contents API (`repos/{owner}/{repo}/contents/{path}`)
- Issues API (`repos/{owner}/{repo}/issues`)
- Pulls API (`repos/{owner}/{repo}/pulls`)
- Releases API (`repos/{owner}/{repo}/releases`)
- Stargazers API (`repos/{owner}/{repo}/stargazers`)
- Contributors API (`repos/{owner}/{repo}/contributors`)

## Tools Pencarian Web:
- **Tavily**: Pencarian terstruktur untuk agen AI dengan hasil kutipan
- **Exa**: Pencarian semantik dengan skor relevansi
- **Linkup**: Analisis AI terintegrasi
- **Firecrawl/FastCRM**: Crawling dan scraping konten web
  - Endpoint: `GET /v1/crawl`, `GET /v1/scrape`
  - Response: HTML bersih atau markdown yang diextract
  - Kelebihan: Ekstraksi konten terstruktur, parsing otomatis
- **Brightdata**: Proxy untuk scraping dengan rate limit
- **Browserbase**: Interactive browser automation

## Tools Artikel Ilmiah:
- **OpenAlex**: Database publikasi terbuka
- **DOI (Crossref)**: Resolusi metadata DOI
- **Scopus API**: Pencarian di database terkurasi

## Tools HTTP & Crawling:
- **curl**: Request HTTP dasar, download raw content dengan response header lengkap
- **FastCRM/Firecrawl**: Crawling dan scraping terstruktur
  - Endpoint: `GET /v1/crawl`, `GET /v1/scrape`
  - Response: HTML bersih atau markdown yang diextract

## Tools FileSystem Tambahan:
- **head**: Baca awal file (efisien untuk file besar)
- **tail**: Baca akhir file (efisien untuk file besar)
