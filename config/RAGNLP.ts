import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Ollama } from "@langchain/ollama";
import { OllamaEmbeddings } from "@langchain/ollama";
import { loadDocumentsJSON, EndpointMetadata } from "../core/retrieval/loaders/loadDocumentJSON3 copy.js";
import { RunnableMap, RunnablePassthrough } from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts";
import path from "path";
import { Document as LangChainDocument } from "langchain/document";

export const config = {
  documentPath: "/home/luca/RagBaseNLP/src/data/", //Cartella con i documenti
  faissIndexPath: "./faiss_index", // Path per salvare l'indice FAISS
  outputPath: "/home/luca/RagBaseNLP/src/response/", // Cartella per salvare le risposte JSON
  modelName: "llama3.2:1b", // Nome modello Ollama
  chunkSize: 1000, // Dimensione chunk per lo splitting
  chunkOverlap: 250, // Overlap tra chunk
  retrievalConfig: {
    k: 15
  },
  jsonSplitting: {
    splitKeys: ['endpoints', 'actions', 'rules'], // Chiavi da splittare
    preserveKeys: ['manifest'] // Chiavi da mantenere intere
  }
};
//llama3.2:3b
export const llm = new Ollama({
  baseUrl: "http://localhost:11434",
  model: config.modelName,
  temperature: 0.01, //Valore più alto = risposte più creative, valore più baso = risposte più concrete
  // format: "json",  //per formato JSon della risposta, se qua formato JSON nel prompt utilizzo JSONOutputParser
  numCtx: 4097, //Aumenta il contesto se possibile
  topP: 0.95 //Per maggiore coerenza
});

export const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",  // URL di Ollama
  model: "nomic-embed-text",
  maxRetries: 2,
  maxConcurrency: 3,//Riduci per concorrenza di debug
});


export interface ExtendDocument extends LangChainDocument {
  metadata: EndpointMetadata,
  readableText?: string; //opzionale per testo leggibile
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


⚠️ USE EXCLUSIVELY THE DATA PROVIDED IN THE GIVEN CONTEXT

You may INTERPRET clearly labeled parameters (e.g., "setpoint" means target temperature) 
but NEVER infer values not present in the data.

⚠️ ALWAYS STATE WHEN INFORMATION IS NOT AVAILABLE IN THE DATA
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



export const targetFile = 'installation-config.json'; //File specifico da processare
export const directoryPath = "/home/luca/RagBaseNLP/src/data";
export const filePath = path.join(directoryPath, targetFile);




//export const user_query = "Give me a list of the uuid from the 'sensor' devices in the configuration. Indicate the name and category.";
//export const user_query = "Dimmi che dispositivi ci sono nel file";
//export const user_query = "Che sensore mi può dare informazioni di temperatura?";
//export const user_query = "Endpoint per il controllo luci";
/***"Come si chiama il termostato"?
"Che sensore mi può dare informazioni di temperatura"? 
"Esistono dei sensori per controllare luci"? */
//export const user_query = "Tell me something about the controller and the firmaware version"; //---> funziona
//export const user_query = "What is the name of thermostat?"
//export const user_query = "Dimmi tutti i parametri del termostato"
//export const user_query = "Give me all parameters of BOX-IO";
//export const user_query = "Show me all the sensors connected to the first floor.";
//export const user_query = "Show me sensors";
//export const user_query = "Show me devices";
//export const user_query = "Accendi le luci";
//export const user_query = "What is the default thermostat setpoint?"; // il "default" rischia di far si che il modello non se la senta di "inventare" perchè non abbiamo un parametro "default" il valore da attribuire
//export const user_query = "What is the value of the thermostat setpoint?";
//export const user_query = "Show me all devices located on the second floor";
//export const user_query = "Show me the UUIDs of actuator, thermostat and controller";
//export const user_query = "Dimmi qual'è l' UUID del controller luci soggiorno";

const user_query = "When the temperature exceeds 25°, turn on the air conditioner"

/**A. Test con Query Tipiche:
- "Mostra i sensori di temperatura"
- "UUID del controller luci soggiorno"  
- "Parametri configurabili termostato"
- "Dispositivi zona cucina" */


/**                           QUERY PER AUTOMAZIONE, DA TESTARE 
 *     "Create an automation for the thermostat"

    "When the temperature exceeds 25°, turn on the air conditioner"

    "Schedule the lights to turn on at 6:00 PM"

    "If there is motion, turn on the lights" 
    
    */



async function main() {
  try {
    await runRgaSytsem(user_query);
  } catch (error) {
    console.log("Errore nell'esecuzione del RAG...");
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

    console.log(response);

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

main();