'use client'

import React from 'react'

type SearchResponse = {
  items: { score: number; meta: Record<string, unknown> }[]
}

export default function SearchPage() {
  const [q, setQ] = React.useState('')
  const [k, setK] = React.useState<number>(5)
  const [loading, setLoading] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [results, setResults] = React.useState<SearchResponse['items']>([])

  async function onSearch(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResults([])
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ q, k }),
      })
      const data = (await res.json()) as SearchResponse | { error?: string }
      if (!res.ok) {
        throw new Error((data as any)?.error || `HTTP ${res.status}`)
      }
      setResults((data as SearchResponse).items || [])
    } catch (err: any) {
      setError(err?.message || 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 800, margin: '32px auto', padding: 16 }}>
      <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 16 }}>Semantic Search</h1>
      <form onSubmit={onSearch} style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Nhập truy vấn..."
          style={{ flex: 1, padding: '8px 12px', border: '1px solid #ccc', borderRadius: 6 }}
        />
        <input
          type="number"
          value={k}
          min={1}
          max={50}
          onChange={(e) => setK(Number(e.target.value))}
          title="Top K"
          style={{ width: 80, padding: '8px 12px', border: '1px solid #ccc', borderRadius: 6 }}
        />
        <button
          type="submit"
          disabled={loading || !q}
          style={{ padding: '8px 16px', borderRadius: 6, border: '1px solid #333', background: '#111', color: '#fff' }}
        >
          {loading ? 'Đang tìm...' : 'Tìm kiếm'}
        </button>
      </form>

      {error && (
        <div style={{ color: '#b91c1c', marginBottom: 12 }}>Lỗi: {error}</div>
      )}

      {results.length > 0 && (
        <div style={{ display: 'grid', gap: 12 }}>
          {results.map((it, idx) => (
            <div key={idx} style={{ border: '1px solid #e5e7eb', borderRadius: 8, padding: 12 }}>
              <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>score: {it.score.toFixed(4)}</div>
              <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {JSON.stringify(it.meta, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


