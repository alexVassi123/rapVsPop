FROM nginx:alpine

RUN rm /etc/nginx/conf.d/default.conf

COPY app/defaultbackend/default.html /usr/share/nginx/html/default.html
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf

