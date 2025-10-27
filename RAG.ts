import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { Ollama } from "@langchain/ollama";
import { promises as fs } from "fs";
import { OllamaEmbeddings } from "@langchain/ollama";
import path from 'path';
import { Document as LangChainDocument } from "langchain/document";
import { loadDocumentsJSON, EndpointMetadata } from "../core/retrieval/loaders/loadDocumentJSON3 copy.js";
import { logChunkSplitting } from "../utils/logging/logChunkSplitting.js";
import { createRagChain } from "../core/chains/Chain.js";
import { ExtendsSeqMetadata } from "src/core/retrieval/splitters/SecondSplit2.js";
import { Chroma } from "@langchain/community/vectorstores/chroma"; //Alternativa a FAISS


export const config = {
  documentPath: "/home/luca/ragts2GeneralQuery New/src/data/", //Cartella con i documenti
  faissIndexPath: "./faiss_index", // Path per salvare l'indice FAISS
  outputPath: "/home/luca/ragts2GeneralQuery New/src/response/", // Cartella per salvare le risposte JSON
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
  format: "json",  //per formato JSon della risposta, se qua formato JSON nel prompt utilizzo JSONOutputParser
  numCtx: 4097, //Aumenta il contesto se possibile
  topP: 0.95 //Per maggiore coerenza
});

export const embeddings = new OllamaEmbeddings({
  baseUrl: "http://127.0.0.1:11434",  // URL di Ollama
  model: "nomic-embed-text",
  maxRetries: 2,
  maxConcurrency: 3,//Riduci per concorrenza di debug
});


//const model = new SentenceTransformer('paraphrase-MiniLM-L6-v2');

export const directoryPath = "/home/luca/ragts2GeneralQuery New/src/data";
//export const user_query = "Give me a list of the uuid from the 'sensor' devices in the configuration. Indicate the name and category.";
//export const user_query = "Dimmi che dispositivi ci sono nel file";
//export const user_query = "Che sensore mi può dare informazioni di temperatura?";
//export const user_query = "Endpoint per il controllo luci";
/***"Come si chiama il termostato"?
"Che sensore mi può dare informazioni di temperatura"? 
"Esistono dei sensori per controllare luci"? */
export const user_query = "Tell me something about the controller and the firmaware version"; //---> funziona
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






/*    1. Query Specifica (Dispositivi)

"Mostrami tutti i termostati presenti nell'impianto con i loro parametri di temperatura"
Verifica:

    Filtraggio preciso per categoria (termostati)

    Estrazione parametri specifici

    Formattazione tabellare

2. Query Generale (Panoramica)

"Descrivi il sistema domotico nel suo complesso"
Verifica:

    Riepilogo non filtrato

    Identificazione categorie principali

    Linguaggio discorsivo

3. Query Comparativa

"Confronta le caratteristiche dei dispositivi BOXIO e WS558"
Verifica:

    Recupero incrociato tra tipologie

    Analisi differenziale

    Tabella comparativa

4. Query Tecnica (Parametri)

"Quali dispositivi misurano il consumo energetico e con quali parametri?"
Verifica:

    Ricerca per parametri specifici

    Mapping dispositivo-parametro

    Precisione tecnica

5. Query Complessa (Multi-filtro)

"Mostra i sensori ambientali nella zona giorno che misurano sia temperatura che umidità"
Verifica:

    Filtri combinati (tipo + posizione + parametri)

    Gestione condizioni complesse

    Rilevanza risultati

Bonus: Query di Fallback

"Cerca dispositivi che non esistono nel sistema" (es. "Mostra le videocamere di sicurezza")
Verifica:

    Gestione errori elegante

    Suggerimenti alternativi

    Riconoscimento limiti del sistema

Ogni query testa un aspetto diverso:

    Precisione (query 1 e 4)

    Sintesi (query 2)

    Analisi (query 3)

    Complessità (query 5)

    Robustezza (bonus)

Per metriche di valutazione, monitora:

    Tempo di risposta

    Numero dispositivi rilevanti restituiti

    Completezza informazioni

    Fluidità linguaggio naturale  */

// export const user_query = "Give me a list of the 'sensor' devices in the configuration. Indicate the name and category.";




//const JsonSchema = await loadSchema(directoryPath);  --> caricamento documento/file Json-Schema

export const DEVICE_CATEGORIES: Record<number, {
  name: string;
  keyParams: string[];
  deviceTypes: string[];
  visualizationTypes: string[];
  description: string;
}> = {
  0: {
    name: 'controller',
    keyParams: ['mac_address', 'firmware_version', 'bsp_version'],
    deviceTypes: ['BOX-IO', 'Thermostat', 'LED Driver', 'Actuator'], // Tutti i dispositivi in categoria 0
    visualizationTypes: ['BOXIO', 'SMABIT_AV2010_32', 'LED_DRIVER', 'GEWISS_GWA1531'], // VisualizationType corrispondenti
    description: 'General control devices'
  },
  /*1: { // categoria per termostati
    name: 'thermostat',
    keyParams: ['temperatura', 'setpoint', 'system_mode'],
    description: 'Termostats and HVAC controls'
  },
  2: { // categoria per attuatori
    name: 'actuator',
    keyParams: ['window_covering_percentage', 'window_covering_command_up'],
    description: 'Actuators and mechanical devices' //Attuatori e dispositivi meccanici
  },*/
  11: { //categoria misuratori di consumo energetico
    name: 'energy_meter',
    keyParams: ['total_active_energy', 'phase_1_current', 'total_system_power'],
    deviceTypes: ['Energy meter'],
    visualizationTypes: ['EASTRON_SDM630'],
    description: 'Energy cosumption meters'
  },
  15: {// categoria per controller per illuminazioni smart
    name: 'smart_light',
    keyParams: ['line_1', 'line_2', 'active_power', 'voltage'],
    deviceTypes: ['Smart lightin controller'],
    visualizationTypes: ['WS558'],
    description: 'Smart lightin controller'
  },

  18: {// categoria sensori ambientali
    name: 'sensor',
    keyParams: ['temperature', 'presence', 'fall'],
    deviceTypes: ['Fall sensor'],
    visualizationTypes: ['VAYYAR_CARE'],
    description: 'Evironmental sensors'
  }
};


/**Ecco la lista completa dei dispositivi presenti con le loro categorie principali:

    BOX-IO (Category: 0) - Dispositivo di gestione generale

    Smart light controller (Category: 15) - Controller per illuminazione

    Thermostat (Category: 1) - Termostato per temperatura

    Line 1 lights (Category: 15) - Luci LED

    Energy meter (Category: 11) - Misuratore di energia elettrica

    Roller shutter actuator (Category: 2) - Attuatore per tapparelle

    Fall sensor (Category: 18) - Sensore di caduta e presenza */





//=================== INTERFACCE PRINCIPALI ========================================


export interface ExtendDocument extends LangChainDocument {
  metadata: EndpointMetadata,
  readableText?: string; //opzionale per testo leggibile
  automationConfig?: AutomationConfig; //L'automatismo configurato con tutti i suoi dettagli
  //È il campo principale che sostituisce il vecchio campo pageContent, e contiene tutta la configurazione 
  // dell'automatismo, come definito nell'interfaccia AutomationConfig.

}

export interface DocumentMetadata {
  // Metadati comuni a tutti i documenti
  source: string;
  loc: string;
  type: 'installation-config' | 'endpoint' | 'area' | 'fallback' | string;
  isValid: boolean;
  timestamp: string;

  // Metadati specifici per endpoint
  uuid?: string;
  name?: string;
  category?: number;
  categoryName?: string;
  isSensor?: boolean;
  visualizationType?: string;
  partitions?: string[];
  parametersCount?: number;
  defaultParameter?: string;

  // Metadati specifici per installation-config
  installationName?: string;
  revision?: string;

}


export interface EndpointSummary {
  uuid: string,
  name: string,
  categoria: string, // Categoria semantica (controller, sensor, actuator)
  location: string,
  configurableParams: string[],
  monitoringParams: string[]
}


export interface StructuredContext {
  metadata: {
    installationName: string;
    revision: string;
  };
  endpoints: EndpointSummary[]; // Lista semplificata di endpoint
  statistics: {
    totalEndpoints: number;
    totalConfigurableParams: number;
  };
}

export interface AutomationConfig {
  manifest: {
    uuid: string;
    name: string;
    description?: string;
    installationUuid: string;
  };
  cron?: {
    expressions: string[];
  };
  inputs: Input[];
  condition: Condition;
  actions: Action[];
  readableText?: string;
}
//Tipi per gli input
type Input = DomoticInput | HealthInput | EventInput;

//Tipi per le azioni
type Action = SetDomoticAction | SendNotificationAction | TriggerAlarmAction;

//Tipi per le condizioni
type Condition = LogicalCondition | ComparisonCondition | TemporalCondition | CronCondition | TriggerCondition | ValueChangeCondition;

// Input interfaces
interface DomoticInput {
  type: "DOMOTIC_PARAMETER";
  source: {
    parameterName: string;
    boxioId: string;
    endpointUuid: string;
  };
}

interface HealthInput {
  type: "HEALTH_PARAMETER";
  source: {
    parameterName: string;
    userId: number;
  };
}

interface EventInput {
  type: "EVENT";
  source: {
    eventType: string;
  };
}

// Action interfaces
interface SetDomoticAction {
  uuid: string;
  type: "SET_DOMOTIC_PARAMETER_VALUE";
  parameter: {
    name: string;
    boxioId: string;
    endpointUuid: string;
  };
  value: string | number | boolean;
}

interface SendNotificationAction {
  uuid: string;
  type: "SEND_NOTIFICATION";
  userIds: number[];
  platforms: ("EMAIL" | "PUSH")[];
  notification: {
    title: string;
    message: string;
  };
}

interface TriggerAlarmAction {
  uuid: string;
  type: "TRIGGER_ALARM";
  userIds: number[];
  alarm: {
    name: string;
    description: string;
    shouldRepeat: boolean;
    repeatInterval: number;
  };
  notification: {
    title: string;
    message: string;
  };
}

// Condition interfaces
interface BaseCondition {
  uuid: string;
  operator: ConditionOperator;
}

type ConditionOperator =
  | "AND"
  | "OR"
  | "GREATER_THAN"
  | "GREATER_THAN_OR_EQUAL"
  | "LESS_THAN"
  | "LESS_THAN_OR_EQUAL"
  | "EQUAL"
  | "MAINTAINED_FOR_SECONDS"
  | "HAS_CHANGED_VALUE"
  | "CRON"
  | "TRIGGER";

interface LogicalCondition extends BaseCondition {
  operator: "AND" | "OR";
  operands: Condition[];
}

interface ComparisonCondition extends BaseCondition {
  operator: "GREATER_THAN" | "GREATER_THAN_OR_EQUAL" | "LESS_THAN" | "LESS_THAN_OR_EQUAL" | "EQUAL";
  operand1: string;
  operand2: string;
}

interface TemporalCondition extends BaseCondition {
  operator: "MAINTAINED_FOR_SECONDS";
  seconds: number;
  condition: Condition;
}

interface CronCondition extends BaseCondition {
  operator: "CRON";
  expressions: string[];
}

interface TriggerCondition extends BaseCondition {
  operator: "TRIGGER";
  operand1: string;
  match: Record<string, unknown>;
}

interface ValueChangeCondition extends BaseCondition {
  operator: "HAS_CHANGED_VALUE";
  operand1: string;
}


//Interfaccia per Parser
// File: types/input.ts
export interface AutomatismContext {
  structure: {
    areas: Area[];
    endpoints: Endpoint[];
  };
  automatism: {
    actions: AutomatismAction[];
    complexRules: ComplexRule[];
    enableAlarms: boolean;
    enableRules: boolean;
    enableScenes: boolean;
  };
  metadata: {
    revision: string;
    structureId: string;
    major: number;
    minor: number;
    name?: string;
  };
}

export interface Area {
  uuid: string;
  name: string;
  partitions: Partition[];
  latitude?: number;
  longitude?: number;
}

interface Partition {
  uuid: string;
  name: string;
}

export interface Endpoint {
  uuid: string;
  name: string;
  category: number;
  parameters: Parameter[];
  partitions: string[];
  visualizationType?: string;
}

export interface Parameter {
  name: string;
  dataType: number;
  unit?: string;
  defaultValue?: string | number | boolean;
  operation?: {
    type: string;
    [key: string]: any;
  }
}

interface AutomatismAction {
  uuid: string;
  name: string;
  operations: Operation[];
  isScene?: boolean;
}

interface Operation {
  parameter: {
    endpoint: string;
    parameter: string;
  };
  value: string;
}

interface ComplexRule {
  uuid: string;
  name: string;
  rules: Condition[]; // Puoi definire un tipo più specifico se necessario
  template?: string;
}

interface RagResponse {
  query: string;
  response: AutomationConfig | string;
  timestamp: string;
  context?: string[];
  validation?: {
    valid: boolean;
    errors?: string[];
  }
}

//=============== Main ========================================================

export const targetFile = 'installation-config.json'; //File specifico da processare
export const filePath = path.join(directoryPath, targetFile);
//const validator = new Validator();

//const outputParser = new JsonOutputParser();


async function debugFAISSIndex(vectorStore: FaissStore) {
  const sampleDocs = await vectorStore.similaritySearch("test", 25);
  // console.log("FAISS Index Metadata Sample:");
  //sampleDocs.forEach((doc, i) => {
  //console.log(`[${i}] Metadata:`, doc.metadata);
  //});
}


/*async function main() {
  try {
    if (!await testOllamaConnection()) {
      throw new Error("Ollama unreachable - start service first");
    }

    console.log("Starting runRAGSystem...");
    const query = user_query;
    const response = await runRAGSystem(query);

    //const JsonSchema = await loadSchema(directoryPath);
    // const validatedResponse = validateLLMResponse(response, directoryPath);


    // Salvataggio della risposta
    await saveResponse(query, response);

    console.log("All completed successfully");




    let parsedResponse;

    if (typeof response === 'string') {
      try {
        parsedResponse = JSON.parse(response);

      } catch (e) {
        console.error("Error parsing LLM response", e);
        return;

      }

    } else {
      parsedResponse = response;
      console.log("Parse response:", parsedResponse);
    }

    //const testEmbeddings = await embeddings.embedQuery("test di prova");
    // console.log("Test Embeddings:", testEmbiddings());

  } catch (error) {
    console.error("Error in MAIN:", {
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : 'No stack trace'
    });
  }
}*/


// METHODS

//===================================== TEST SERVER CONNECTION ===============================================
async function testOllamaConnection() {
  try {
    const test = await fetch('http://localhost:11434', {
      method: 'GET'
    });
    console.log("Connection OK, status:", test.status);
    return true;
  } catch (error) {
    console.error("Connection failed:", error);
    return false;
  }
}

// ===================================== CORE RUG FUNCTIONS ================================================

export async function runRAGSystem(query: string): Promise<string | AutomationConfig> {



  try {

    //Gestione errore da parte di Ollama:
    /**
     * async function runRAGSystem(query: string, documents: any[]) {
    try {
        // Aggiungi timeout alla chiamata Ollama
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 secondi timeout

        const response = await ollamaModel.invoke(query, {
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        return response;

    } catch (error) {
        if (error.name === 'AbortError') {
            console.error('❌ Timeout: Ollama non ha risposto entro 30 secondi');
        } else {
            console.error('❌ Errore Ollama:', error.message);
        }
        
        // Fallback: ritorna una risposta di default
        return {
            text: "Mi dispiace, il servizio di intelligenza artificiale non è al momento disponibile. " +
                  "Riprova più tardi o verifica che Ollama sia in esecuzione."
        };
    }
}
     */
    if (!query || typeof query !== 'string') {
      throw new Error(`Not valid query: ${typeof query} - ${query}`);
    }

    console.time('RAG_System_Total_Time');

    console.time('Loading_Documents');

    //    const Documents = await loadDocumentsJSON();

    const { Documents, partitionMap } = await loadDocumentsJSON();


    console.timeEnd('Loading_Documents');

    const globalPartitionMap = partitionMap;

    // 2. Verifica che sia popolata
    console.log("PartitionMap size:", globalPartitionMap.size);
    for (const [uuid, name] of globalPartitionMap.entries()) {
      console.log(`PartitionMap entry: ${uuid} -> ${name}`);
    }


    console.log("Original documents loaded. Checking 'category' fields:");

    Documents.forEach((d: any, idx: any) => {

      try {
        const device = JSON.parse(d.pageContent)?.device;
        if (!device) console.log(`[Doc ${idx}] Nessun campo device trovato`);
      } catch (e) {
        console.log(`[Doc ${idx}] JSON parse error`);
      }
    });


    // Document Validation
    const validDocuments = Documents.filter((doc: any) =>
      doc.pageContent && typeof doc.pageContent === 'string'
    );

    if (validDocuments.length === 0) {
      throw new Error("No valid JSON documents found");
    }


    //const globalPartitionMap = globalPartitionMap;


    // 4. CREATION CHUNKS
    console.time('Creating_Chunks');


    //if (allChunks.length === 0) {
    //  console.error("No chunks available: please verify that the data has been loaded correctly");
    // throw new Error("Vectore store not initialized: chunks.length ===  0");
    // }

    console.timeEnd('Creating_Chunks');

    //allChunks.slice(0, 3).forEach((chunk, i) => {
    // console.log(`Chunk ${i}: ${chunk.metadata.uuid || 'no-uuid'}`);
    // });

    console.time('Chunk_Validation');
    const depth = 5;
    const chunkingResults = await logChunkSplitting(validDocuments, config.chunkSize, config.chunkOverlap, depth); //config.chunkSize, config.chunkOverlap); //Documents
    console.timeEnd('Chunk_Validation')

    // 5. Vectore Store Management
    console.time('Vector_Store_Management');
    let vectorStore: FaissStore;



    //DEBUGGG


    // Estrai TUTTI i chunk splittati da tutti i risultati
    let allSplittedChunks = chunkingResults.flatMap(result => result.documents);
    console.log(`   - Total splitted chunks: ${allSplittedChunks.length}`);


    // ⚠️ NUOVO DEBUG: Verifica dimensioni chunk prima dell'embedding
    console.log(`📊 CHUNK SIZE ANALYSIS BEFORE EMBEDDING:`);
    const chunkSizes = allSplittedChunks.map(chunk => chunk.pageContent.length);
    console.log(`   - Total chunks: ${allSplittedChunks.length}`);
    console.log(`   - Max chunk size: ${Math.max(...chunkSizes)} chars`);
    console.log(`   - Min chunk size: ${Math.min(...chunkSizes)} chars`);
    console.log(`   - Avg chunk size: ${Math.round(chunkSizes.reduce((a, b) => a + b) / allSplittedChunks.length)} chars`);

    // Trova i chunk troppo grandi
    const oversizedChunks = allSplittedChunks.filter(chunk => chunk.pageContent.length > 4000);
    if (oversizedChunks.length > 0) {
      console.warn(`🚨 ${oversizedChunks.length} CHUNKS STILL OVERSIZED FOR EMBEDDING:`);
      oversizedChunks.forEach(chunk => {
        console.warn(`   - ${chunk.metadata.name || 'unnamed'}: ${chunk.pageContent.length} chars`);
        console.warn(`     Type: ${chunk.metadata.chunkType}, UUID: ${chunk.metadata.uuid}`);
      });

      // ⚠️ FIX IMMEDIATO: Filtra i chunk troppo grandi
      const safeChunks = allSplittedChunks.filter(chunk => chunk.pageContent.length <= 4000);
      console.log(`⚠️ Using ${safeChunks.length} safe chunks instead of ${allSplittedChunks.length}`);
      allSplittedChunks = safeChunks; // Sostituisci con chunk sicuri
    }




    try {

      // Try loading the existing index
      vectorStore = await FaissStore.load(config.faissIndexPath, embeddings);
      console.log("FIASS index loaded from disk");

      await debugFAISSIndex(vectorStore);
      // Verificare il contenuto dell'indice


    } catch (error) {
      console.warn("Faiss index not found, creating a new index...");

      vectorStore = await FaissStore.fromDocuments(allSplittedChunks, embeddings);//allChunks ma cambiando logica non ha più senso, se loadDoc carica chunks


      // In RAG.ts, when saving to FAISS:
      console.log("Saving chunks to FAISS with metadata:");
      allSplittedChunks.forEach((chunk, index) => {
        console.log(`Chunk ${index}: ${chunk.metadata.chunkType}, isFirstFloor: ${chunk.metadata.isFirstFloor}.isSecondFloor: ${chunk.metadata.isSecondFloor}, hasAreaInfo: ${chunk.metadata.hasAreaInfo}`);
      });

      await vectorStore.save(config.faissIndexPath);

      console.log("New FAISS index created and saved");
    }

    console.timeEnd('Vector_Store_Management');


    // Debug: conta tutti i documenti nel vectorStore
    /* try {
 
       const allDocs = await vectorStore.similaritySearch("", 10000); // Query vuota, limite alto
       console.log(`Debug: VectorStore contiene ${allDocs.length} documenti totali`);
       //allDocs.forEach((doc, i) => {
       //  console.log(`[${i}] ${doc.metadata.chunkType}: ${doc.metadata.areaName || doc.metadata.name}`);
       //});
     } catch (error) {
       console.warn("Impossibile contare i documenti nel vectorStore:", error);
     }*/

    //DEBUGGGGGGGGGGGGGGGGGGGGGGGGGG    
    // Prima di chiamare prepareDocumentsForTwoStage
    console.log("Documents before two-stage preparation:");
    Documents.forEach((doc, i) => {
      console.log(`[${i}] ${doc.metadata?.chunkType}: ${doc.pageContent?.substring(0, 1000)}...`);
    });



    //FAISS NON SUPPORTA FILTRI CHE UTILIZZANO METADATI, NEANCHE CHIAVE,VALORE, 

    const k = config.retrievalConfig.k;


    /**💡 Attenzione: similaritySearch() è immediato ma non ha re-ranking interno (ad es. Max Marginal Relevance), e se query === "", restituisce solo i più simili a niente (quindi comporta fallback). */
    // Re-index solo i documenti validi in un nuovo vector store
    // const tempVectorStore = await FaissStore.fromDocuments(relevantDocs, new OpenAIEmbeddings());

    const retriever = vectorStore.asRetriever({
      k,
      searchType: "similarity",//'mmr',
      //searchKwargs: { lambda: 0.5, fetchK: 20 },
      verbose: true
    })

    let retrievalDocs = await retriever.invoke(query);
    console.log("All available chunks:");

    //Without manual filtering, beacuse it's applied in twoStageRetrieval

    retrievalDocs.forEach(doc => {
      console.log(`- ${doc.metadata.name}: chunkType=${doc.metadata.chunkType} pageContent:${doc.pageContent.length} visualizationTypes:${doc.metadata.visualizationType} - categories: ${doc.metadata.category}`);
    });

    /**🧠 Usa asRetriever() se:
    
    Vuoi miglior ranking (searchType: "mmr", ad es.)
    
    Vuoi passare un retriever a un chain o agent
    
    Vuoi riusare il retriever più volte */


    console.log("Search for relevant documents...");
    retrievalDocs.forEach((doc, i) => {
      console.log(`- [${i}] ${doc.metadata.name} (UUID: ${doc.metadata.uuid}) 
          - type ${doc.metadata.type} - chunkType: ${doc.metadata.chunkType} 
          - floorLocation: ${doc.metadata.floorLocation} - Qualcosa: ${doc.metadata.floorInfo}
          .IsFirstFloor: ${doc.metadata.isFirstFloor} -isSecondFloor: ${doc.metadata.isSecondFloor}`);

    });
    console.log("retrievalDocs dimension No filtering from", retrievalDocs.length);


    // const items = faiss_index.similaritySearch("Che dispositivo richiede la query?", k = 100);

    retrievalDocs = prioritizeChunks(retrievalDocs);


    console.log("Controlliamo ora cosa c'è in retrievalDocs:", retrievalDocs.length);
    //console.log("Controlliamo ora cosa c'è in retrievalDocs:", JSON.stringify(retrievalDocs));

    //const finalDocs = filterAndDeduplicateDocuments(retrievalDocs, query);
    //console.log(`Final documents after filtering: ${finalDocs.length}`);




    // Extract query embedding
    const queryEmbedding = await embeddings.embedQuery(query); //=============> Inutile, solo curiosità per vedere come viene trasformata una stringa in embeddings

    //console.log("QueryEmbeddings", JSON.stringify(queryEmbedding));


    //  Creating the chain with LCEL
    console.log("Chain creation RAG...");
    const chain = await createRagChain(llm, vectorStore, retrievalDocs);//scoredDocs);//finalDocs provo ad utilizzare i documenti filtrati dal coseno similiraity!!

    //  Invoking the chain with the query
    const response = await chain.invoke({ query });

    console.timeEnd('RAG_System_Total_Time');

    return response;

  } catch (error) {
    console.error("System erro RAG:", error);
    return await handleSystemError(query, error);
  }

}

/* ====================================================================================
    FUNCTION FOR PRIORITIZING CHUNKS 
   ==========================================================================================
*/

function prioritizeChunks(docs: LangChainDocument[]): LangChainDocument[] {
  return docs.sort((a, b) => {
    const aMeta = a.metadata as ExtendsSeqMetadata;
    const bMeta = b.metadata as ExtendsSeqMetadata;

    if (aMeta.sequenceNumberSystem?.chunkId !== undefined && bMeta.sequenceNumberSystem?.chunkId !== undefined) {


      if (aMeta.sequenceNumberSystem?.chunkId < 0 && bMeta.sequenceNumberSystem?.chunkId < 0) return -1;

      //1. Parent chunks FIRST(highest priority - chunkId = 0)
      if (aMeta.sequenceNumberSystem?.chunkId === 0 && bMeta.sequenceNumberSystem?.chunkId !== 0) return -1;
      if (bMeta.sequenceNumberSystem?.chunkId === 0 && aMeta.sequenceNumberSystem?.chunkId !== 0) return -1;

      //2. Non-SequenceNumber chunks SECOND (medium priority)
      if (!aMeta.sequenceNumberSystem && bMeta.sequenceNumberSystem) return -1;
      if (!bMeta.sequenceNumberSystem && aMeta.sequenceNumberSystem) return -1;

      //3. Child chunks LAST (Lowest priority) - mantain their order
      if (aMeta.sequenceNumberSystem?.chunkId > 0 && bMeta.sequenceNumberSystem?.chunkId > 0) {
        return aMeta.sequenceNumberSystem?.chunkId - bMeta.sequenceNumberSystem?.chunkId;
      }

      return 0;
    }
    return -1;

  });

}



async function handleSystemError(query: string, error: any): Promise<string> {
  console.error("🚨 System error managment:", error);

  return JSON.stringify({
    response: "An error occured while processing your request.",
    error: error instanceof Error ? error.message : String(error),
    query: query,
    timestamp: new Date().toISOString()
  }, null, 2);
}








export function convertLangChainDocs(docs: any[]): LangChainDocument[] {
  //se relevantDocs[0]  non ha pageContent, ma ha un campo simile (es. content o text), modifica convertLangChainDocs per mapparlo correttamente:
  //docs è un'array 
  console.log("Pre-conversion content:", docs[0]);
  return docs.map(doc => {
    //Controllo se il documento ha già la strottura ExtendDocument
    if (doc.pageContent && doc.metadata) {
      return doc as LangChainDocument;
    }

    return {
      pageContent: doc.pageContent || doc.content || "",
      metadata: doc.metadata || {
        source: 'unknown',
        loc: 'unknown',
        section: doc.metadata?.section || 'no-section'

      }
    };
  });

}

//Function for texting how the embiddings works
/*
async function testEmbiddings() {
  const testQueries = [
    "motion sensor input",
    "ligth control action",
    "temperature condition"
  ];
  for (const query of testQueries) {
    const embedding = await embeddings.embedQuery(query);
    console.log(`Embedding for "${query}":`, {
      length: embedding.length,
      first10: embedding.slice(0, 10)
    });
    // Test similarity with known terms
    const docEmbedding = await embeddings.embedQuery("DOMOTIC_PARAMETER");
    const similarity = cosineSimilarity(embedding, docEmbedding);
    console.log(`Similarity with "DOMOTIC_PARAMETER": ${similarity}`);
  }
}
*/



/**
 * Cosine similarity support function
 * 
 * For additional filter after retrieval. */
function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));

  console.log("Cosine similarity:", JSON.stringify(dotProduct / (magnitudeA * magnitudeB)));
  return dotProduct / (magnitudeA * magnitudeB);
}


//shows the actions already defined or available, useful for understanding "what it can do".
export function getActionsSummary(structuredContext: AutomatismContext): string {
  if (!structuredContext || !structuredContext.automatism?.actions) {
    return "No actions available";
  }

  return structuredContext.automatism.actions.map(action => {
    const operations = action.operations
      .map(op => `${op.parameter.parameter}=${op.value}`)
      .join(', ') || "No operation ";

    return `- ${action.name} (${action.uuid}): ${operations}`;
  }).join('\n');
}


export function estimateTokenCount(text: string) {
  //Stima conservativa: 1 toke = 4 caratteri 
  return Math.ceil(text.length / 4);
}

/*      NOT BAD FUNCTION, REPAIR THE TRUNCATED JSON, BUT NOT RELEVANT NOW
function fixTruncatedJson(truncated: string): string {
  try {
    //Tentativo 1: aggiungo chiusura mancante
    if (!truncated.trim().endsWith('}') && !truncated.trim().endsWith(']')) {
      const lastBrace = Math.max(
        truncated.lastIndexOf('{'),
        truncated.lastIndexOf('[')
      );
      if (lastBrace > 0) {
        return truncated.slice(0, lastBrace) +
          (truncated[lastBrace] === '{' ? '}' : ']');
      }
    }
    // Tentativo 2: Estrai l'oggetto più interno completo
    const jsonObjects = truncated.match(/{[^{}]*}|\[[^\[\]]*\]/g);
    if (jsonObjects && jsonObjects.length > 0) {
      return jsonObjects[jsonObjects.length - 1];
    }
    throw new Error('JSON impossible to repair');
  } catch (e) {
    console.error('JSON repair failed, using fallback:', e);
    return '{}'; // Fallback minimale
  }
}*/



// ======================================= SAVE THE RESPONSE (JSON FORMAT) ====================================
async function saveResponse(query: string, response: string | any): Promise<void> {

  const responseText = typeof response === 'string' ? response : JSON.stringify(response, null, 2);


  console.log("Start saveResponse");

  try {
    // Create a folder if it doesn't exist yet
    await fs.mkdir(config.outputPath, { recursive: true });

    // Create a unique file name
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `response_${timestamp}.json`;
    const fullPath = path.join(config.outputPath, filename);

    // Data to save
    /*const dataToSave = {
      query,
      response,
      timestamp: new Date().toISOString(),
      model: config.modelName
    };
    const ragResponse: RagResponse = {
      query,
      response,
      timestamp: new Date().toISOString(),
      context: [] // Aggiungi il contesto se necessario  ------------------>???????
    };*/

    // Save the file
    await fs.writeFile(
      fullPath,
      responseText,
      //JSON.stringify(ragResponse, null, 2),
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