FROM nginx:alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx/nginx.conf /etc/nginx/conf.d
COPY ../app/frontend/genre-classifier/dist/ /usr/share/nginx/html
