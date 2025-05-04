import { GoogleGenerativeAI } from "@google/generative-ai";

// Validate API key
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
    throw new Error("GOOGLE_API_KEY environment variable is not set");
}

export async function queryLLM(query: string, data: { documents: string[], metadata: any[] }) {
    console.log('🤖 Initializing Gemini...');
    try {
        const genAI = new GoogleGenerativeAI(GEMINI_API_KEY || "");
        // Using pro model instead of flash for better stability
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

        console.log('📝 Preparing prompt with context...');
        const prompt = `Based on the following website content, answer the question.
The content is from multiple pages of the website:

${data.metadata.map((meta, i) => 
    `Page ${i + 1}: ${meta.title}
URL: ${meta.url}
Content:
${data.documents[i]}
---`
).join('\n\n')}

Question: ${query}

Please provide a clear and structured answer based on the content above. If listing pages, include their titles and URLs.`;

        console.log('🔄 Generating response...');
        const result = await model.generateContent(prompt);
        if (!result.response) {
            throw new Error("No response received from Gemini");
        }
        console.log('✅ Response generated successfully');
        return result.response.text();
    } catch (error: any) {
        console.error("❌ Error querying Gemini:", {
            message: error.message,
            status: error.status,
            details: error.errorDetails || 'No additional details'
        });
        if (error.status === 403) {
            throw new Error("Invalid or missing API key. Please check your GOOGLE_API_KEY environment variable.");
        }
        throw error;
    }
}