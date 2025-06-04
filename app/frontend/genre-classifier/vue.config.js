module.exports = {
  // Base URL (adjust if deploying to subdirectory)
  publicPath: '/',
  
  devServer: {
    proxy: {
      '/api': {  // Proxy all requests starting with '/api'
        target: 'http://localhost:8000',  // Your FastAPI backend
        changeOrigin: true,
        pathRewrite: {
          '^/api': ''  // Remove '/api' prefix when forwarding to FastAPI
        }
      }
    }
  }
}