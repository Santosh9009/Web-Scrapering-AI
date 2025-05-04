import { Pipeline } from '@xenova/transformers';
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import { ChromaClient, IncludeEnum } from 'chromadb';

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
            console.log('📝 Attempting to get existing collection...');
            const collection = await client.getCollection({ 
                name: collectionName,
                embeddingFunction
            });
            console.log('✅ Retrieved existing collection');
            return collection;
        } catch (error: any) {
            // If collection doesn't exist, create it
            if (error.message?.includes('Collection not found')) {
                console.log('📝 Collection not found, creating new one...');
                const collection = await client.createCollection({ 
                    name: collectionName,
                    embeddingFunction
                });
                console.log('✅ Created new collection');
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

async function initializeEmbeddingPipeline() {
    if (!embeddingPipeline) {
        console.log('🔄 Initializing embedding pipeline...');
        const { pipeline: transformerPipeline } = await import('@xenova/transformers');
        embeddingPipeline = await transformerPipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            quantized: false
        }) as Pipeline;
        console.log('✅ Embedding pipeline initialized');
    }
    return embeddingPipeline;
}

async function getEmbeddings(text: string): Promise<number[]> {
    try {
        const pipeline = await initializeEmbeddingPipeline();
        const output: { data: Float32Array } = await pipeline(text, { 
            pooling: 'mean', 
            normalize: true 
        });
        return Array.from(output.data);
    } catch (error) {
        console.error('❌ Error in getEmbeddings:', error);
        throw error;
    }
}

// Embedding function wrapper for ChromaDB
const embedder = {
    embedQuery: getEmbeddings,
    embedDocuments: async (texts: string[]) => {
        // Initialize pipeline once before processing batch
        await initializeEmbeddingPipeline();
        // Process texts sequentially to avoid memory issues
        const embeddings = [];
        for (const text of texts) {
            const embedding = await getEmbeddings(text);
            embeddings.push(embedding);
        }
        return embeddings;
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
        
        // Process chunks in smaller batches
        const batchSize = 5; // Reduced batch size
        console.log(`🔄 Processing ${docs.length} documents in batches of ${batchSize}...`);
        
        for (let i = 0; i < docs.length; i += batchSize) {
            const batch = docs.slice(i, i + batchSize);
            console.log(`📊 Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(docs.length / batchSize)}...`);
            
            const embeddings = await embedder.embedDocuments(
                batch.map(doc => doc.pageContent)
            );
            console.log(`✅ Generated embeddings for batch ${Math.floor(i / batchSize) + 1}`);
            
            console.log(`💾 Storing batch ${Math.floor(i / batchSize) + 1} in ChromaDB...`);
            await collection.add({
                ids: batch.map(() => crypto.randomUUID()),
                embeddings,
                metadatas: batch.map(doc => ({ url, timestamp: new Date().toISOString() })),
                documents: batch.map(doc => doc.pageContent)
            });
            
            console.log(`✅ Stored batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(docs.length / batchSize)}`);
            
            // Add a small delay between batches to prevent overwhelming the system
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        console.log('✅ All documents stored successfully!');
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
            nResults: 10,
            where: { url: url },  
            include: [IncludeEnum.Metadatas, IncludeEnum.Documents]
        });

        console.log('results of searchRelevantData are :'+ JSON.stringify(results))
        
        const documents = results.documents[0].filter((doc): doc is string => 
            doc !== null && typeof doc === 'string'
        );
        const metadata = results.metadatas[0].filter(meta => meta !== null);
        
        console.log(`✅ Found ${documents.length} relevant documents for URL: ${url}`);
        return {
            documents: documents.slice(0, 3),
            metadata: metadata.slice(0, 3)
        };
    } catch (error) {
        console.error('❌ Error searching data:', error);
        throw error;
    }
}

export async function getStoredDocuments(url?: string) {
    try {
        console.log('🔄 Retrieving stored documents...');
        
        if (!url) {
            console.log('⚠️ No URL provided');
            return {
                total: 0,
                url: null,
                documents: []
            };
        }

        const collection = await getOrCreateCollection();
        console.log('📚 Querying documents for URL:', url);
        
        try {
            const results = await collection.get({
                where: { url: url },
                include: [IncludeEnum.Metadatas, IncludeEnum.Documents]
            });

            console.log(`✅ Found ${results.ids.length} documents for URL: ${url}`);
            
            return {
                total: results.ids.length,
                url: url,
                documents: results.documents.map((doc, i) => ({
                    id: results.ids[i],
                    content: doc,
                    metadata: results.metadatas[i]
                }))
            };
        } catch (error) {
            console.log('⚠️ No documents found for URL:', url);
            return {
                total: 0,
                url: url,
                documents: []
            };
        }
    } catch (error) {
        console.error('❌ Error retrieving documents:', error);
        throw error;
    }
}
