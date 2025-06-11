FROM node:20-alpine as builder

WORKDIR /app

COPY app/frontend/genre-classifier/ .

RUN npm install && npm run build

FROM nginx:alpine

RUN rm /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html/home

COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

