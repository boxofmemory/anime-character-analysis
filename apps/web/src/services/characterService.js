import { http } from './http'

export async function renameStranger(oldName, newName) {
  const { data } = await http.post('/api/characters/rename-stranger', {
    old_name: oldName,
    new_name: newName,
  })
  return data
}
