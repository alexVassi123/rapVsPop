# Build Stage
FROM node:20-alpine as build

# Set working directory
WORKDIR /app

# Copy Vue project files
COPY app/frontend/genre-classifier/ .

# Install dependencies and build
RUN npm install && npm run build

# Production Stage
FROM nginx:alpine

# Remove default NGINX config (optional but recommended if using custom config)
RUN rm /etc/nginx/conf.d/default.conf

# Copy built frontend from previous stage
COPY --from=build /app/dist /usr/share/nginx/html

# Optional: Add custom NGINX config if needed
COPY ../nginx/nginx.conf /etc/nginx/conf.d/default.conf
