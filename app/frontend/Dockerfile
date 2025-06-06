FROM node:20-alpine as build
WORKDIR /app
COPY app/frontend/genre-classifier/ .
RUN npm install && npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
