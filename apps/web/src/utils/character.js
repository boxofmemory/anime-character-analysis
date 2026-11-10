export const nameDict = {
  Aquamarine_Hoshino: '阿库亚',
  Hoshino_Ruby: '星野露比',
  Hoshino_Ai: '星野爱',
}

export function translateName(name) {
  if (!name) return ''
  if (name.startsWith('Stranger_')) {
    return name.replace('Stranger_', '未知角色_')
  }
  return nameDict[name] || name
}

export function getCharacterDurations(timelineMap) {
  return Object.entries(timelineMap)
    .map(([name, info]) => {
      const total = info.segments.reduce((sum, segment) => sum + (segment.end - segment.start), 0)
      return { name, total: Number(total.toFixed(2)), type: info.type }
    })
    .sort((a, b) => b.total - a.total)
}
