FROM node:16-alpine as build

WORKDIR /app

# Install curl for healthcheck
RUN apk --no-cache add curl

# Copy package.json and install dependencies
COPY package*.json ./
RUN npm install --production

# Copy the rest of the code and build
COPY . .
RUN npm run build

# Production environment
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
