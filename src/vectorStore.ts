import { Pipeline } from '@xenova/transformers';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import { ChromaClient } from 'chromadb';

// Initialize ChromaDB client
const client = new ChromaClient({
    path: "http://localhost:8000"
});
const collectionName = "websites";

// Create ChromaDB compatible embedding function
const embeddingFunction = {
    generate: async (texts: string[]) => {
        return await embedder.embedDocuments(texts);
    }
};

// Add collection management
async function getOrCreateCollection() {
    try {
        console.log('🔄 Getting ChromaDB collection...');
        
        try {
            // Try to get existing collection first
            const collection = await client.getCollection({ 
                name: collectionName,
                embeddingFunction
            });
            console.log('✅ Found existing collection');
            return collection;
        } catch (error: any) {
            // If collection doesn't exist, create it
            if (error.message?.includes('Collection not found')) {
                console.log('📝 Creating new collection...');
                const collection = await client.createCollection({ 
                    name: collectionName,
                    embeddingFunction
                });
                return collection;
            }
            throw error;
        }
    } catch (error) {
        console.error('❌ ChromaDB collection error:', error);
        throw error;
    }
}

// Initialize the embedding pipeline
let embeddingPipeline: Pipeline | null = null;

async function getEmbeddings(text: string): Promise<number[]> {
    if (!embeddingPipeline) {
        const { pipeline: transformerPipeline } = await import('@xenova/transformers');
        embeddingPipeline = await transformerPipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            quantized: false
        }) as Pipeline;
    }
    const output: { data: Float32Array } = await embeddingPipeline(text, { 
        pooling: 'mean', 
        normalize: true 
    });
    return Array.from(output.data);
}

// Embedding function wrapper for ChromaDB
const embedder = {
    embedQuery: getEmbeddings,
    embedDocuments: async (texts: string[]) => {
        return Promise.all(texts.map(text => getEmbeddings(text)));
    }
};

export async function storeEmbeddings(text: string, url: string) {
    try {
        console.log('📑 Splitting text into chunks...');
        const splitter = new RecursiveCharacterTextSplitter({ 
            chunkSize: 500,
            chunkOverlap: 50 
        });
        const chunks = await splitter.splitText(text);
        console.log(`📚 Created ${chunks.length} text chunks`);

        console.log('🔄 Creating document objects...');
        const docs = chunks.map(chunk => 
            new Document({ 
                pageContent: chunk,
                metadata: { url, timestamp: new Date().toISOString() }
            })
        );

        console.log('💾 Storing documents in ChromaDB...');
        const collection = await getOrCreateCollection();
        
        // Process chunks in batches to avoid overwhelming the server
        const batchSize = 10;
        for (let i = 0; i < docs.length; i += batchSize) {
            const batch = docs.slice(i, i + batchSize);
            const embeddings = await embedder.embedDocuments(
                batch.map(doc => doc.pageContent)
            );
            
            await collection.add({
                ids: batch.map(() => crypto.randomUUID()),
                embeddings,
                metadatas: batch.map(doc => ({ url, timestamp: new Date().toISOString() })),
                documents: batch.map(doc => doc.pageContent)
            });
            
            console.log(`✅ Stored batch ${i / batchSize + 1} of ${Math.ceil(docs.length / batchSize)}`);
        }

        return collection;
    } catch (error) {
        console.error('❌ Error storing embeddings:', error);
        throw error;
    }
}

export async function searchRelevantData(query: string, url: string) {
    try {
        console.log('🔄 Connecting to ChromaDB...');
        const collection = await getOrCreateCollection();
        
        console.log('🔍 Searching for relevant documents...');
        const queryEmbedding = await embedder.embedQuery(query);
        
        const results = await collection.query({
            queryEmbeddings: [queryEmbedding],
            nResults: 10, // Increase results to ensure we find matches for the URL
            where: { url: url }, // Filter by URL
        });
        
        // Filter out null values and ensure we have an array of strings
        const documents = results.documents[0].filter((doc): doc is string => 
            doc !== null && typeof doc === 'string'
        );
        
        console.log(`✅ Found ${documents.length} relevant documents for URL: ${url}`);
        return documents.slice(0, 3); // Return top 3 most relevant results
    } catch (error) {
        console.error('❌ Error searching data:', error);
        throw error;
    }
}
