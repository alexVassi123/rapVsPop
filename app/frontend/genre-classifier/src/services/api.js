import axios from 'axios';



const apiClient = axios.create({
  baseURL: '/home/api',
  withCredentials: false,
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
  }
});

export default {
  predictGenre(lyrics) {
    return apiClient.post('/predict', { lyrics });
  },
  healthCheck() {
    return apiClient.get('/health');
  }
}
