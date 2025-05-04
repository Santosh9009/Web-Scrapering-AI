import 'dotenv/config';
import express from "express";
import cors from "cors";
import path from "path";
import { fileURLToPath } from 'url';
import { scrapeWebsite } from "./scraper.js";
import { storeEmbeddings, searchRelevantData, getStoredDocuments } from "./vectorStore.js";
import { queryLLM } from "./llm.js";
import dotenv from "dotenv";

dotenv.config();


const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// Endpoint to scrape and store website data
app.post("/api/scrape", async (req, res) => {
    const { url, config } = req.body;
    console.log(`[${new Date().toISOString()}] 🌐 Scraping request received for URL:`, url);
    try {
        console.log('📑 Starting web scraping with config:', config);
        const scrapedPages = await scrapeWebsite(url);
        const scrapedText = scrapedPages
            .map(page => `${page.title}\n${page.content}`)
            .join('\n=== New Page ===\n');
        
        console.log(`📦 Scraped content length: ${scrapedText.length} characters`);
        
        console.log('💾 Storing embeddings...');
        await storeEmbeddings(scrapedText, url);
        console.log('✅ Embeddings stored successfully');
        
        res.json({ 
            success: true, 
            message: "Website content stored successfully",
            stats: {
                characters: scrapedText.length,
                pages: scrapedPages.length
            }
        });
    } catch (error) {
        console.error(`❌ Scraping failed for URL ${url}:`, error);
        res.status(500).json({ error: "Failed to scrape website" });
    }
});

// Endpoint to query the stored data
app.post("/api/query", async (req, res) => {
    const { query, url } = req.body;
    console.log(`[${new Date().toISOString()}] ❓ Query received for URL ${url}:`, query);
    try {
        console.log('🔍 Searching relevant data...');
        const relevantData = await searchRelevantData(query, url);
        console.log(`📝 Found ${relevantData.documents.length} relevant segments`);
        
        console.log('🤖 Querying LLM...');
        const llmResponse = await queryLLM(query, relevantData);
        console.log('✅ LLM response received');
        
        res.json({ answer: llmResponse });
    } catch (error) {
        console.error('❌ Query processing failed:', error);
        res.status(500).json({ error: "Failed to process query" });
    }
});

// Endpoint to retrieve stored documents
app.get("/api/stored-data", async (req, res) => {
    const url = req.body.url as string | undefined;
    console.log(`[${new Date().toISOString()}] 📚 Retrieving stored data${url ? ` for URL: ${url}` : ''}`);
    
    try {
        const data = await getStoredDocuments(url);
        res.json({
            success: true,
            ...data
        });
    } catch (error) {
        console.error('❌ Failed to retrieve stored data:', error);
        res.status(500).json({ error: "Failed to retrieve stored data" });
    }
});

app.listen(3000, () => console.log("Server running on port 3000"));
