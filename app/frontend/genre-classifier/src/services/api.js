import axios from 'axios';

const apiClient = axios.create({
  baseURL: process.env.NODE_ENV === 'development' 
    ? 'http://localhost:8000' 
    : '/api', // When deployed, Nginx will handle /api
  withCredentials: false,
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
  }
});

export default {
  predictGenre(lyrics) {
    return apiClient.post('/api/predict', { lyrics });
  },
  healthCheck() {
    return apiClient.get('/health');
  }
}
