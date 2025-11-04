import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Ollama } from "@langchain/ollama";
import { OllamaEmbeddings } from "@langchain/ollama";
import { loadDocumentsJSON, EndpointMetadata } from "../core/retrieval/loaders/loadDocumentJSON.js";
import { RunnableMap, RunnablePassthrough } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import path from "path";
import { Document as LangChainDocument } from "langchain/document";
import { user_query } from "./Query.js";
import { promises as fs } from "fs";


export const config = {
  documentPath: "/home/luca/RagBaseNLP/src/data/", // Folder containing the documents
  faissIndexPath: "./faiss_index", // Path to save the FAISS index
  outputPath: "/home/luca/RagBaseNLP/src/response/", // Folder to save JSON responses
  modelName: "llama3.2:1b", // Ollama model name
  chunkSize: 1000, // Chunk size for text splitting
  chunkOverlap: 250, // Overlap between chunks
  retrievalConfig: {
    k: 15
  },
  jsonSplitting: {
    splitKeys: ['endpoints', 'actions', 'rules'], // Keys to split
    preserveKeys: ['manifest'] // Keys to keep intact
  }
};

// llama3.2:3b
export const llm = new Ollama({
  baseUrl: "http://localhost:11434",
  model: config.modelName,
  temperature: 0.01, // Higher value = more creative responses, lower value = more factual answers
  // format: "json",  // Use this for JSON output format; if enabled, use JSONOutputParser in the prompt
  numCtx: 4097, // Increase context size if possible
  topP: 0.95 // For greater coherence
});

export const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",  // Ollama API URL
  model: "nomic-embed-text",
  maxRetries: 2,
  maxConcurrency: 3, // Reduce this if debugging concurrency issues
});

export interface ExtendDocument extends LangChainDocument {
  metadata: EndpointMetadata,
  readableText?: string; // Optional field for human-readable text
}


const prompt = PromptTemplate.fromTemplate(`
  You are an assistant specialized in the analysis of home automation systems. Your task is to provide accurate information based solely on the data provided in the context.
It is essential that you maintain a strictly analytical approach and never add information that is not explicitly present in the data available to you.

IMPORTANT! 
Responde in Natural Language, discorsive.

CORE OPERATING PRINCIPLES

Your primary responsibility is to be a faithful interpreter of the provided data. 
This means you must always base your answers on the concrete facts present in the context, avoiding any form of speculation,
 logical deduction, or addition of information based on general knowledge. 
If a piece of information is not explicitly present in the data, it is your responsibility to clearly communicate this to the user.

But if the context is very verbose, use the data that is most relevant to you for that particular query, For example, if the query requires only one parameter, return that parameter without returning all the others.


USE EXCLUSIVELY THE DATA PROVIDED IN THE GIVEN CONTEXT

You may INTERPRET clearly labeled parameters (e.g., "setpoint" means target temperature) 
but NEVER infer values not present in the data.

ALWAYS STATE WHEN INFORMATION IS NOT AVAILABLE IN THE DATA
AVAILABLE DEVICES CONTEXT

{context}
USER REQUEST

{query}
ANALYSIS METHODOLOGY
Preliminary Verification Phase

Before formulating any answer, you must conduct a systematic analysis of the available data. Start by identifying whether the requested device or information is actually present in the provided context. Never assume a device exists just because the user mentions it—always verify its presence in the data.

Carefully examine the structure of the data to understand which parameters are actually documented and which values are specified. Remember that the absence of a parameter in the data does not mean it doesn't exist in the real system, but simply that you do not have sufficient information to discuss it.
Information Extraction Process

Once the requested device is verified to be present, proceed with the information extraction following a rigorous methodology. Focus exclusively on what is literally present in the data, using the exact field names and specified values.

From the context provided to you, only use the devices requested by the query!
If the query required the all parameters of devices, give me all technical details of every parameter. If the query required some parameters, show pnly the paramters reuired.
Communication of Limitations

It is essential that you always communicate the limitations of the information at your disposal. Users must understand that your analysis is based on the specific data provided and that additional information not visible in the current context may exist.


  ANALYSIS RULES:
  1. NEVER list individual parameters as separate devices
  2. ALWAYS group parameters under their parent device
  3. If a parameter chunk is found, find its parent device first
  4. Ignore parameter chunks that don't belong to any parent

  `);



export const targetFile = 'installation-config.json'; //Specific file to process
export const directoryPath = "/home/luca/RagBaseNLP/src/data";
export const filePath = path.join(directoryPath, targetFile);



/*
   ============================================================================================================
                                                    MAIN
   ============================================================================================================
*/

async function main() {
  try {
    let response = await runRgaSytsem(user_query);


    // Salvataggio della risposta
    await saveResponse(user_query, response);

  } catch (error) {
    console.log("Error while running RAG...");
  }
}

async function runRgaSytsem(query: string) {

  try {

    const Document = loadDocumentsJSON();

    //Create a Vector Store and load FAISS index
    let vectoreStore: FaissStore;
    vectoreStore = await FaissStore.load(config.faissIndexPath, embeddings);

    let k = config.retrievalConfig.k;



    //  retrievalDocs = await vectoreStore.similaritySearch(user_query, config.retrievalConfig.k);

    /**Versione con asRetrieval
     **/
    const retriever = vectoreStore.asRetriever({
      k,
      searchType: "similarity", //Non supportrato "mmr"
      verbose: true
    });

    const retrievalDocs = await retriever.invoke(user_query);
    console.log("Quanti documenti ha recuperato l'asRetriever? Ecco qua:", retrievalDocs.length);

    const filteringDocs = filterByContentRelevance(retrievalDocs, user_query, 0.3);

    console.log("Quanti documenti otteniamo dopo il filtro per rilevanza?", filteringDocs.length);

    const context = filteringDocs.map((doc: any) => doc.pageContent).join("\n\n");




    /*const chain = {
      {
      "context": retrievalDocs.map(doc => doc.pageContent).join("\n\n"),
        "query": RunnablePassthrough.from(async (input: { query: string }) => input.query),
        
      }
      | prompt={ `You are an expert system for home automation configuration. Using the provided context, answer the following query in a concise manner.` }
        | model: llm
      | StringOutputParser
    */

    const chain = RunnableMap.from({
      context: () => context,
      query: new RunnablePassthrough()//RunnablePassthrough.isRunnable((input: { query: string }) => input.query),
    })
      .pipe(prompt)
      .pipe(llm);

    const response = await chain.invoke({ query: "When the temperature exceeds 25°, turn on the air conditioner" });

    console.log("\n\nQuestion:", user_query);
    console.log("\nAnswer:", response);


    return response;

  } catch (error) {
    console.log("Error in RAG system:", error);
    return { system: "error" };
  }

}



function filterByContentRelevance(docs: string | any, query: string, threshold = 0.3) {
  const queryTerms = query.toLowerCase().split(/\s+/).filter(term => term.length > 2);

  return docs.filter((doc: any) => {
    const content = doc.pageContent.toLowerCase();
    let score = 0;

    queryTerms.forEach(term => {
      if (content.includes(term)) score += 1;
    });

    // Normalizza il score per lunghezza query
    const relevanceScore = score / queryTerms.length;
    return relevanceScore >= threshold;
  });
}



//  ===========================================================================
//                        SAVE THE RESPONSE (NLP FORMAT)
//  ===========================================================================
async function saveResponse(query: string, response: string | any): Promise<void> {

  //const responseText = typeof response === 'string' ? response : JSON.stringify(response, null, 2);

  console.log("Start saveResponse");

  try {
    // Create a folder if it doesn't exist yet
    await fs.mkdir(config.outputPath, { recursive: true });

    // Create a unique file name
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `response_${timestamp}.md`; //saving in .md format
    const fullPath = path.join(config.outputPath, filename);

    // Save the file
    await fs.writeFile(
      fullPath,
      //responseText,
      response,
      "utf-8"
    );

    console.log("Response saved in:", fullPath);
    console.log("Query executed: ", query);
  } catch (error) {
    console.error("Error saving response:", error);
    throw error;
  }
}

main();