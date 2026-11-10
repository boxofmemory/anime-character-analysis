import { http } from './http'

export async function getHistory() {
  const { data } = await http.get('/api/history')
  return data
}
