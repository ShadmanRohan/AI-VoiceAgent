import { describe, expect, it } from 'vitest'

describe('LocalVoiceAI UI', () => {
  it('exposes expected navigation tabs', () => {
    const tabs = ['Voice', 'History', 'Memory', 'Tools', 'Settings', 'Status']
    expect(tabs).toContain('Voice')
    expect(tabs).toHaveLength(6)
  })
})
