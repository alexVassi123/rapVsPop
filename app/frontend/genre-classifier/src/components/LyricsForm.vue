<template>
  <div class="lyrics-form">
    <h1>Lyrics Classifier</h1>
    <textarea v-model="lyrics" placeholder="Paste lyrics here..." rows="8"></textarea>
    <button @click="submitLyrics">Classify</button>

    <p v-if="loading">Loading...</p>
    <p v-if="result">Prediction: <strong>{{ result }}</strong></p>
  </div>
</template>

<script>
import { classifyLyrics } from "@/services/api";

export default {
  data() {
    return {
      lyrics: "",
      result: null,
      loading: false,
    };
  },
  methods: {
    async submitLyrics() {
      this.loading = true;
      this.result = null;
      try {
        const response = await classifyLyrics(this.lyrics);
        this.result = response.label;
      } catch (err) {
        this.result = "Error: Could not classify";
        console.error(err);
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style scoped>
.lyrics-form {
  max-width: 600px;
  margin: 4rem auto;
  padding: 2rem;
  background-color: #f9f9f9;
  border-radius: 12px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  text-align: center;
}

.lyrics-form h1 {
  font-size: 2rem;
  margin-bottom: 1.5rem;
  color: #333;
}

textarea {
  width: 100%;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #ccc;
  resize: vertical;
  font-size: 1rem;
  margin-bottom: 1rem;
  box-sizing: border-box;
}

button {
  background-color: #4caf50;
  color: white;
  border: none;
  border-radius: 8px;
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

button:hover {
  background-color: #45a049;
}

p {
  margin-top: 1rem;
  font-size: 1.1rem;
  color: #444;
}

strong {
  color: #222;
}
</style>
