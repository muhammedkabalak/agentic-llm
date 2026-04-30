# Node.js tabanını alıyoruz
FROM node:18-alpine

WORKDIR /app

# Paket dosyalarını kopyalayıp kuruyoruz
COPY package*.json ./
RUN npm install

# Tüm arayüz kodlarını kopyalıyoruz
COPY . .

# Dışarıya açılacak port (Vite genelde 5173 kullanır)
EXPOSE 5173

# Frontend'i dış ağa açarak başlatıyoruz
CMD ["npm", "run", "dev", "--", "--host"]