
import { ExtendDocument, targetFile, filePath, DEVICE_CATEGORIES } from "../../../config/RAG.js";
import { filterByVisualizationType } from "../../chains/Chain.js"
import { promises as fs } from "fs";
import { Document } from "langchain/document";//Document non è un array e non ha un metodo push
import { getConfig } from "src/services/query/getConfig.js";
import { buildGlobalPartitionMap } from "./buildGlobalPartitionsMap.js";
import { getCategoryName, getDeviceType } from "../splitters/createSchemaAware3.js"
import { createSchemaAwareChunk } from "../splitters/createSchemaAware3.js";


/*
Ora viene creato un unico documento che contiene:

    Tutti gli endpoint processati in un array

    Tutte le aree processate in un array

    Statistiche complete dell'installazione

    Mappa globale delle partizioni*/





// Document è una classe, non un array, Gli array hanno .push(), ma le classi no, Document è un costruttore di oggetti, non un array


//codice TypeScript per l’arricchimento semantico di dati JSON in un contesto RAG (Retrieval-Augmented Generation)


export type DocumentType = 'installation-config';
export type ChunkType = "summary" | "detail" | "area" | "fallback";


export interface EndpointMetadata {
    //Metadata di base
    source: string;  //Source file name
    loc: string;  //Full file path
    type: DocumentType // Type document
    isValid: boolean;
    timestamp: string;
    chunkType: ChunkType,  // Chunks type (summary, detail, area, fallback)

    //Dati endpoint OBBLIGATORI
    uuid?: string,   //device uuid
    name?: string,   //decvice name
    category?: number,    //category (0, 15, 11,18)
    visualizationType?: string,  //Visualization type(BOXIO, VAYYAR_CARE, etc)

    //Dati OPZIONALI
    categoryName?: string,    //Category name mapped
    partitions?: string[],    //UUID of associated partitions
    location?: string[],
    areaNames?: string[], //Areas names
    areaUuids?: string[], //UUID of the areas   
    id?: string,
    parametersCount?: number,  //Parameters number  
    defaultParameter?: string,  //Parametro predefinito


    // ========================================
    // METADATI SPECIFICI PER TWO-STAGE CHUNKS
    // ========================================
    isPrimaryChunk?: boolean,
    chunkStrategy?: 'two_stage' | 'standard';  // identifies the chunking strategy used


    parameterStartIndex?: number,
    parameterEndIndex?: number,
    totalParameters?: number,

    visualizationCategory?: string;  // Mapped View Category

    deviceType: string,
    globalPartitionMapArray?: Array<[string, string]>;


    // Flags per il filtering
    hasAreaInfo?: boolean;
    hasEndpoints?: boolean;
    hasConfiguration?: boolean;
    hasControlParams?: boolean;
    isFirstFloor?: string;
    isSecondFloor?: string;

    // Per area chunks
    areaName?: string;
    areaUuid?: string;
    areaIndex?: number;
    devicesCount?: number;
    floorName?: string;
    deviceTypes?: string[];
    deviceCategories?: number[];
    partitionNames?: string[];
    partitionIds?: string[];

    // Per detail chunks
    isSensor?: boolean;
    isActuator?: boolean;
    isController?: boolean;
    hasMeasurementParameters?: boolean;
    hasEnumerationParameters?: boolean;
    hasConfigParameters?: boolean;
    parameterUnits?: string[];
    parameterDataTypes?: string[];

    // Per splitting
    subChunkIndex?: number;
    totalSubChunks?: number;
    splitField?: string;
    fullUuid?: string;
    isSubChunk?: boolean;
    warning?: string;
    error?: string;


    parameterNames?: string[];
    parameterOperations?: string[];
    hasMeasurementParams?: boolean;

    //revision?: string;
    //major?: number;
    //minor?: number;

    totalEndpoints?: number;
    totalAreas?: number;
    hasPartitions?: boolean; //Indicates whether the document has partitions
    installationName?: string;
    revision?: string,
    minor?: number;
    major?: number;

    [key: string]: any;

    sequenceNumberMetadata?: SeqMetadata;   //---> VIENE CALCOLATO DURANTE LO SPLITTING, COME FACCIO A DEFINIRLO QUA NELLA CREAZIONE DEI CHUNKS
}

export interface SeqMetadata {
    sessionId: string;
    chunkId: number;
    totalChunks: number;
    parentChunkId?: number;
    isParent?: boolean;
    isAckChunk?: boolean;
}

export interface Parameter {
    name: string,
    dataType: number,
    unit?: string,
    operation?: { type: string },
    logType?: number,
    defaultStateValue?: string,
    notifyFrequency?: number,
    maxVal: number[],
    minVal: number[],
    [key: string]: any,


}

// NUOVE INTERFACCE PER MIGLIORARE LA MAPPATURA
export interface AreaPartitionMap {
    areaUuid: string;
    areaName: string;
    partitions: Array<{
        uuid: string;
        name: string;
    }>;
}

export interface EndpointAreaRelation {
    endpointUuid: string;
    endpointName: string;
    areaUuid: string;
    areaName: string;
    partitionUuids: string[];
    location: string[];
}




export async function loadDocumentsJSON(): Promise<{ Documents: ExtendDocument[], partitionMap: Map<string, any> }> {
    const processedUUIDs = new Set<string>();
    let rawContent: string;

    try {
        rawContent = await fs.readFile(filePath, 'utf-8');
    } catch (error) {
        console.error("ERROR: Document not found or not readable!");
        throw new Error("Execution blocked: file doesn't exist");
    }

    if (!rawContent || rawContent.trim().length === 0) {
        console.error("ERROR: Empty document!");
        throw new Error("Execution blocked: file contents empty.");
    }

    let jsonContent: any;
    try {
        jsonContent = JSON.parse(rawContent);
    } catch (parseError) {
        console.error("ERROR: JSON parsing failed", parseError);
        throw new Error("Invalid JSON format in configuration");
    }

    if (!jsonContent || typeof jsonContent !== 'object') {
        throw new Error("Invalid JSON structure: root must be an object");
    }

    try {

        const hasValidEndpoints = Array.isArray(jsonContent.endpoints) && jsonContent.endpoints.length > 0;
        const hasValidAreas = Array.isArray(jsonContent.areas) && jsonContent.areas.length > 0;

        if (!hasValidEndpoints) {
            console.warn("No valid endpoint found in JSON content");
            return getFallbackDocument(new Error("No valid endpoints in configuration"));
        }

        console.log(`Data structure: ${jsonContent.endpoints.length} endpoints, ${jsonContent.areas?.length || 0} area`);

        // Build global maps
        const globalPartitionMap = buildGlobalPartitionMap(jsonContent);
        const areaPartitionMaps = hasValidAreas ? buildAreaPartitionMaps(jsonContent) : [];
        const endpointAreaRelations = hasValidAreas ? buildEndpointAreaRelations(jsonContent, areaPartitionMaps) : new Map<string, EndpointAreaRelation>();

        // Create installation content
        const installationContent = {
            type: "installation-config",
            metadata: jsonContent.metadata || {},
            statistics: {
                totalEndpoints: jsonContent.endpoints?.length || 0,
                totalAreas: jsonContent.areas?.length || 0,
                totalPartitions: globalPartitionMap.size,
                sensorCount: jsonContent.endpoints.filter((ep: any) => ep.category === 18).length || 0,
                actuatorCount: jsonContent.endpoints?.filter((ep: any) => [11, 12, 15].includes(ep.category)).length || 0,
                controllerCount: jsonContent.endpoints?.filter((ep: any) => [0, 1, 2].includes(ep.category)).length || 0
            },
            endpoints: jsonContent.endpoints,
            areas: jsonContent.areas,
            globalPartitionMap: Object.fromEntries(globalPartitionMap)
        };

        const installationMetadata: EndpointMetadata = {
            source: targetFile,
            loc: filePath,
            type: 'installation-config',
            isValid: true,
            timestamp: new Date().toISOString(),
            chunkStrategy: 'standard',
            isPrimaryChunk: true,
            chunkType: 'summary',
            uuid: 'installation-config',
            name: jsonContent.metadata?.name || 'Installation Config',
            category: -1, //-------------------------------------------------------> perchè -1?
            visualizationType: 'INSTALLATION',
            deviceType: 'installation',
            totalEndpoints: jsonContent.endpoints?.length || 0,
            totalAreas: jsonContent.areas?.length || 0,
            hasPartitions: globalPartitionMap.size > 0,
            hasAreaInfo: hasValidAreas,
            installationName: jsonContent.metadata?.name,
            revision: jsonContent.metadata?.revision,
            major: jsonContent.metadata?.major,
            minor: jsonContent.metadata?.minor,

        };

        const Documents: ExtendDocument[] = [];

        // Main installation document (summary)
        const installationDoc = new Document({
            pageContent: JSON.stringify({
                type: "installation-config",
                chunkType: "summary",
                metadata: jsonContent.metadata,
                statistics: installationContent.statistics
            }),
            metadata: installationMetadata
        }) as ExtendDocument;
        Documents.push(installationDoc);






        // 2. Area documents
        if (jsonContent.areas && Array.isArray(jsonContent.areas)) {
            jsonContent.areas.forEach((area: any) => {
                /*const partitionNames = area.partitions?.map((p: any) =>
                    p.name.toLowerCase()?.includes('first') ? 'First Floor' :
                        p.name.toLowerCase()?.includes('second') ? 'Second Floor' :
                            p.name || 'Unknown Partition'
                ) || []; */

                // CORREZIONE: Migliora la logica di rilevamento dei piani mantenendo il tipo stringa
                const partitionNames = area.partitions?.map((p: any) =>
                    typeof p === 'object' ? p.name : String(p)) || [];

                const isFirstFloor = partitionNames?.some((name: string) =>
                    name && (name.toLowerCase().includes('first') ||
                        name.toLowerCase().includes('primo'))
                    //p?.name?.toLowerCase().includes('First') || p?.name?.toLowerCase().includes('primo')
                );
                const isSecondFloor = partitionNames.some((name: string) =>//= area.partitions?.some((name: string) =>
                    name && (name.toLowerCase().includes('second') ||
                        name.toLowerCase().includes('secondo'))
                    // p?.name?.toLowerCase().includes('Second') || p?.name?.toLowerCase().includes('secondo')
                );
                const areaDoc = new Document({
                    pageContent: JSON.stringify({
                        type: "installation-config",
                        chunkType: "area",
                        area: area,
                        associatedEndpoints: jsonContent.endpoints?.filter((ep: any) =>
                            ep.partitions?.some((part: any) =>
                                area.partitions?.some((areaPart: any) =>
                                    areaPart.uuid === part || areaPart === part
                                )
                            )
                        )
                    }),
                    metadata: {
                        source: targetFile,
                        loc: filePath,
                        type: 'installation-config',
                        isValid: true,
                        timestamp: new Date().toISOString(),
                        chunkStrategy: 'standard',
                        isPrimaryChunk: true,
                        chunkType: 'area',
                        uuid: area.uuid, // Use area UUID for consistency
                        name: area.name,
                        //category: -1,
                        visualizationType: 'AREA',
                        deviceType: 'area',
                        areaUuid: area.uuid,
                        areaName: area.name,


                        isFirstFloor: isFirstFloor ? "first" : "Unknown",
                        isSecondFloor: isSecondFloor ? "second" : "Unknown",
                        floorNames: area.partitions?.map((p: any) => p?.name)
                            .filter((name: string) => name?.toLowerCase()?.includes('floor')) || [], /*filter((name: string) =>
                                name.toLowerCase().includes('floor')
                            ),*/
                        hasAreaInfo: true, // Explicitly set to true
                        partitions: area.partitions?.map((p: any) => ({
                            uuid: p.uuid,
                            name: p.name,
                            isFirstFloor: p.name.toLowerCase().includes('first') ? "first" : "Unknown",
                            isSecondFloor: p.name.toLowerCase().includes('Second') ? "Second" : "Unknown"
                        })) || [],

                        partitionNames: area.partitions?.map((p: any) => p.name || `Partition_${p.uuid?.substring(0, 8)}`) || [],
                        parametersCount: 0,
                        defaultParameter: '',
                    }
                });
                console.log("Areas from JSON:", jsonContent.areas);
                Documents.push(areaDoc as ExtendDocument);
            });
        }

        // 3. Endpoint documents
        if (jsonContent.endpoints && Array.isArray(jsonContent.endpoints)) {

            jsonContent.endpoints.forEach((endpoint: any) => {

                const partitionInfo = (endpoint.partitions || []).map((uuid: string) => {
                    const partition = globalPartitionMap.get(uuid);
                    const partitionName = partition || `Unknown_${uuid.substring(0, 8)}`;
                    //DEBUG, controllo che partition sia una stringa valida
                    const lowerName = partitionName.toLowerCase();

                    return {
                        uuid: uuid,
                        name: partitionName,//|| `Unknown_${uuid.substring(0, 8)}`,  //---> Ritorno un oggetto! 
                        isFirstFloor: lowerName.includes('first') || lowerName.includes('primo') ? "first" : "Unknown",
                        isSecondFloor: lowerName.includes('second') || lowerName.includes('secondo') ? "second" : "Unknown"
                    };
                });

                /*        IL PROBLEMA È CHE QUA LI STO RICALCOLANDO, VOGLIO UTILIZZARE QUELLI GIÀ RICALCOLATI
                const isFirstFloor = partitionInfo.some((partitionName: string) =>
                    partitionName.toLowerCase().includes('first') ||
                    partitionName.toLowerCase().includes('primo')
                );
                const isSecondFloor = partitionInfo.some((partitionName: string) =>
                    partitionName.toLowerCase().includes('second') ||
                    partitionName.toLowerCase().includes('secondo')
                );*/

                const isFirstFloor = partitionInfo.some((p: any) => p.isFirstFloor === "first");
                const isSecondFloor = partitionInfo.some((p: any) => p.isSecondFloor === "second");


                const endpointDoc = new Document({
                    pageContent: JSON.stringify({
                        type: "installation-config",
                        chunkType: "detail" as ChunkType,
                        endpoint: endpoint
                    }),
                    metadata: {
                        //...installationMetadata,
                        source: targetFile,
                        loc: filePath,
                        type: 'installation-config',
                        isValid: true,
                        timestamp: new Date().toISOString(),

                        chunkType: 'detail' as ChunkType,
                        uuid: endpoint.uuid,
                        name: endpoint.name,
                        category: endpoint.category,
                        categoryName: getCategoryName(endpoint.category),
                        deviceType: getDeviceType(endpoint.category),
                        visualizationType: endpoint.visualizationType,
                        partitions: endpoint.partitions || [],
                        partitionInfo: partitionInfo,
                        partitionNames: (endpoint.partitions || []).map((uuid: string) => {
                            const name = globalPartitionMap.get(uuid);
                            return name || `Unknown_${uuid.substring(0, 8)}`;
                        }),
                        // Area references
                        areaUuid: jsonContent.areas?.find((area: any) =>
                            area.partitions?.some((areaPart: any) =>
                                endpoint.partitions?.includes(
                                    typeof areaPart === 'object' ? areaPart.uuid : areaPart
                                )
                            )
                        )?.uuid,
                        areaName: jsonContent.areas?.find((area: any) =>
                            area.partitions?.some((areaPart: any) =>
                                endpoint.partitions?.includes(
                                    typeof areaPart === 'object' ? areaPart.uuid : areaPart
                                )
                            )
                        )?.name,
                        isFirstFloor: isFirstFloor ? "first" : "Unknown",
                        isSecondFloor: isSecondFloor ? "second" : "Unknown",
                        floorLocation: isFirstFloor ? "first" : isSecondFloor ? "second" : "unknown",
                        hasAreaInfo: (endpoint.partitions || []).length > 0,
                        installationName: jsonContent.metadata?.name,
                        revision: jsonContent.metadata?.revision,
                        major: jsonContent.metadata?.major,
                        minor: jsonContent.metadata?.minor
                    }
                });
                Documents.push(endpointDoc as ExtendDocument);
            });
        }

        console.log("Single installation-config document created without chunking");
        //return [installationDoc];
        return {
            Documents,
            partitionMap: globalPartitionMap
        };

    } catch (error) {
        console.error("Critical error loading:", error);
        return getFallbackDocument(error);
    }
}

// ========================================
// FUNCTIONS HELPER FOR TWO-STAGE
// ========================================
export function prepareDocumentsForTwoStage(documents: ExtendDocument[]): { parsedContent: any, metadata: EndpointMetadata }[] {

    console.log("=== PREPARING DOCUMENTS FOR TWO-STAGE ===");
    console.log(`Input: ${documents.length} pre-created chunks`);

    if (!Array.isArray(documents)) {
        console.error(" Input documents is not valid array");
        return [];
    }

    const parsedDocs: { parsedContent: any, metadata: EndpointMetadata }[] = [];
    const chunkTypeStats: Record<string, number> = {};
    const locationStats = { firstFloor: 0, secondFloor: 0, both: 0, neither: 0 };


    for (const [index, doc] of documents.entries()) {
        try {
            // ROBUST DOCUMENT VALIDATION
            if (!doc) {
                console.warn(` Document ${index} is null/undefined`);
                continue;
            }

            if (!doc.metadata) {
                console.warn(` Document ${index} without metadata`);
                continue;
            }

            if (doc.metadata.isValid === false) {
                console.warn(` Document ${index} marked as invalid`);
                continue;
            }

            if (!doc.pageContent || typeof doc.pageContent !== 'string') {
                console.warn(` Document ${index} without valid pageContent`);
                continue;
            }

            // Parse the content - these are already structured chunks
            let parsedContent: any;
            try {
                parsedContent = JSON.parse(doc.pageContent);
            } catch (parseError) {
                console.error(` Error parsing document ${index} content:`, parseError);
                continue;
            }

            if (!parsedContent || typeof parsedContent !== 'object') {
                console.warn(` Parsed content ${index} is not valid object`);
                continue;
            }

            // Normalizza la struttura: assicurati che chunkType sia consistente
            if (parsedContent.chunkType && doc.metadata.chunkType !== parsedContent.chunkType) {
                console.warn(`Chunk type mismatch: metadata=${doc.metadata.chunkType}, content=${parsedContent.chunkType}. Fixing...`);

                // Mantieni il chunkType originale dai metadati invece del contenuto
                parsedContent.chunkType = doc.metadata.chunkType;
            }

            // Estrai le aree dalla struttura corretta
            if (parsedContent.chunkType === 'area' && parsedContent.area) {
                parsedContent.areas = [parsedContent.area]; //areas top level, area inside areas
                delete parsedContent.area;
            }

            // Estrai gli endpoint dalla struttura corretta
            if (parsedContent.chunkType === 'detail' && parsedContent.endpoint) {
                parsedContent.endpoints = [parsedContent.endpoint];
                delete parsedContent.endpoint;
            }

            /*
            parsedDocs.push({
                parsedContent: parsedContent,
                metadata: doc.metadata as EndpointMetadata
            }); */

            parsedDocs.push({ parsedContent, metadata: doc.metadata });

            // RACCOGLI STATISTICHE E DEBUG
            const chunkType = doc.metadata.chunkType;
            chunkTypeStats[chunkType] = (chunkTypeStats[chunkType] || 0) + 1;

            // DEBUG AREA CHUNKS
            if (chunkType === 'area') {
                const isFirst = doc.metadata.isFirstFloor;
                const isSecond = doc.metadata.isSecondFloor;

                if (isFirst && isSecond) locationStats.both++;
                else if (isFirst) locationStats.firstFloor++;
                else if (isSecond) locationStats.secondFloor++;
                else locationStats.neither++;

                console.log(`✅ Area chunk: ${doc.metadata.areaName}, First: ${isFirst}, Second: ${isSecond}`);
            }



            //DEBUGGGGG
            try {
                parsedContent = JSON.parse(doc.pageContent);
            } catch (parseError) {
                console.error(`❌ Error parsing document ${index} content:`, parseError);
                console.error(`   Content: "${doc.pageContent.substring(0, 100)}..."`); // Log del contenuto problematico
                continue;
            }

        } catch (error) {
            console.error(` Error processing document ${index}:`, error);
            continue;
        }
    }

    // FINAL VALIDATION AND STATISTICS
    console.log(`Documents prepared: ${parsedDocs.length}/${documents.length} valid chunks`);

    if (parsedDocs.length === 0) {
        console.error(" No valid documents prepared for two-stage");
        return [];
    }

    // Statistics by chunk type
    console.log("Chunk type distribution:");
    Object.entries(chunkTypeStats).forEach(([type, count]) => {
        console.log(`   • ${type}: ${count}`);
    });

    // Location-based statistics (critical for the reported issue)
    console.log("Location statistics (area chunks only):");
    console.log(`   • First floor: ${locationStats.firstFloor}`);
    console.log(`   • Second floor: ${locationStats.secondFloor}`);
    console.log(`   • Both floors: ${locationStats.both}`);
    console.log(`   • No floor info: ${locationStats.neither}`);

    // Validation warnings
    const hasAreaChunks = parsedDocs.some(doc => doc.metadata.chunkType === 'area');
    const hasSummaryChunks = parsedDocs.some(doc => doc.metadata.chunkType === 'summary');
    const hasDetailChunks = parsedDocs.some(doc => doc.metadata.chunkType === 'detail');

    console.log(" Chunk type validation:");
    console.log(`   • Summary chunks: ${hasSummaryChunks ? '✅' : '❌'}`);
    console.log(`   • Area chunks: ${hasAreaChunks ? '✅' : '❌'}`);
    console.log(`   • Detail chunks: ${hasDetailChunks ? '✅' : '❌'}`);

    if (!hasAreaChunks) {
        console.warn(" WARNING: No area chunks found! This will break location-based queries like 'first floor sensors'");
    }

    // Critical validation for the reported bug
    const firstFloorAreaChunks = parsedDocs.filter(doc =>
        doc.metadata.chunkType === 'area' && doc.metadata.isFirstFloor === 'first'
    );

    if (firstFloorAreaChunks.length === 0) {
        console.warn(" CRITICAL: No area chunks with isFirstFloor=true found!");
        console.warn("   This matches the reported bug where first floor area chunks are missing");
    } else {
        console.log(` Found ${firstFloorAreaChunks.length} first-floor area chunks - should fix the reported issue`);
    }

    console.log("=== TWO-STAGE PREPARATION COMPLETED ===\n");


    console.log(" Validating area chunks for location metadata...");
    const areaChunks = parsedDocs.filter(doc => doc.metadata.chunkType === 'area');
    areaChunks.forEach((doc, i) => {
        console.log(`Area ${i}: ${doc.metadata.areaName} - first: ${doc.metadata.isFirstFloor}, second: ${doc.metadata.isSecondFloor}`);
    });

    const missingLocationMetadata = areaChunks.filter(doc =>
        doc.metadata.isFirstFloor === undefined && doc.metadata.isSecondFloor === undefined
    );

    if (missingLocationMetadata.length > 0) {
        console.warn(`⚠️ ${missingLocationMetadata.length} area chunks missing location metadata!`);
    }




    return parsedDocs;
}

/*   if (!Array.isArray(documents)) {
       console.error(" Input documents is not valid array");
       return [];
   }
 
   const parsedDocs = documents
       .filter((doc, index) => {
           // ROBUST DOCUMENT VALIDATION
           if (!doc) {
               console.warn(` Document ${index} is null/undefined`);
               return false;
           }
 
           if (!doc.metadata) {
               console.warn(` Document ${index} without metadata`);
               return false;
           }
 
           if (!doc.metadata.isValid) {
               console.warn(` Document ${index} marked as invalid`);
               return false;
           }
 
           if (!doc.pageContent || typeof doc.pageContent !== 'string') {
               console.warn(` Document ${index} without pageContent valido`);
               return false;
           }
 
           return true;
       })
       .map((doc, index) => {
           try {
               let parsedContent: any;
               try {
                   parsedContent = JSON.parse(doc.pageContent);
                   // console.log("Vediamo che cosa c'è in doc.pageContent--->", JSON.parse(doc.pageContent));
               }
               catch (parseError) {
                   console.error(`❌ Errore parsing documento ${index}:`, parseError);
                   return null;
               }
 
               if (!parsedContent || typeof parsedContent !== 'object') {
                   console.warn(`⚠️ Documento ${index}: contenuto parsato non valido`);
                   return null;
               }
               //Normalizzation data structure based on type
               const normalizedContent = {
                   ...parsedContent,
                   //For installation-config documents
                   ...(parsedContent.type === 'installation-config' && {
                       endpoints: parsedContent.overview?.endpoints || [],
                       areas: parsedContent.overview?.areas || [],
                       statistics: parsedContent.statistics || {}
                   }),
 
                   /*   // For endpoint documents
                      ...(parsedContent.type === 'endpoint' && {
                          data: parsedContent.data || parsedContent,
                          parameters: parsedContent.data?.parameters || parsedContent.parameters || [],
                          device: parsedContent.device
                      }),
                      // For area documents
                      ...(parsedContent.type === 'area' && {
                          data: parsedContent.data || parsedContent,
                          devices: parsedContent.data?.devices || parsedContent.devices || [],
                          partitions: parsedContent.data?.partitions || parsedContent.partitions || []
                      })        /
 
               };
 
               return {
                   parsedContent: normalizedContent,
                   metadata: doc.metadata as EndpointMetadata
               };
 
           } catch (error) {
               console.error(`Error during parsing ${index}: `, error);
               return null;
           }
 
       })
       .filter((doc): doc is { parsedContent: any, metadata: EndpointMetadata } => {
           //TYPE GUARD for TypeScript
           return doc !== null && doc !== undefined;
       });
   console.log(`Documents prepared: ${parsedDocs.length}/${documents.length} for two-stage`);
 
   // FINAL VALIDATION 
   if (parsedDocs.length === 0) {
       console.error(" No valid documents prepared for two-stage");
   }
   // Statistiche per tipo di documento
   const stats = parsedDocs.reduce((acc, doc) => {
       const type = doc.metadata.type;
       acc[type] = (acc[type] || 0) + 1;
       return acc;
   }, {} as Record<string, number>);
   console.log(" Statistics of prepared documents:", stats);
 
 
 
   return parsedDocs;*/



// ========================================
// FUNZIONE DI VALIDAZIONE PER DEBUG 
// ========================================

export function validateDocumentsForTwoStage(parsedDocs: { parsedContent: any, metadata: EndpointMetadata }[]): boolean {




    console.log("===  DOCUMENT VALIDATION PER TWO-STAGE ===");

    if (!parsedDocs || parsedDocs.length === 0) {
        console.error(" No documents to validate");
        return false;
    }

    let isValid = true;
    let endpointCount = 0;
    let installationCount = 0;
    let areaCount = 0;
    let documentsWithPartitions = 0;
    let documentsWithAreas = 0;

    for (const [index, doc] of parsedDocs.entries()) {
        // Contatori per statistiche

        if (doc.metadata?.type === 'installation-config') installationCount++;
        if (doc.metadata?.partitions !== undefined && doc.metadata?.partitions?.length > 0) documentsWithPartitions++;
        if (doc.metadata?.areaNames !== undefined && doc.metadata?.areaNames?.length > 0) documentsWithAreas++;

        // Validazioni critiche
        if (!doc.parsedContent) {
            console.error(` Document ${index}: parsedContent missing`);
            isValid = false;
        }

        if (!doc.metadata) {
            console.error(` Document ${index}: metadata missing`);
            isValid = false;
        }

        if (doc.parsedContent?.type === 'endpoint' && !doc.parsedContent?.data) {
            console.error(` Document ${index}: endpoint without data`);
            isValid = false;
        }

        // Log dettagliato solo per documenti problematici
        if (!doc.parsedContent || !doc.metadata ||
            (doc.parsedContent?.type === 'endpoint' && !doc.parsedContent?.data) ||
            (doc.parsedContent?.type === 'area' && !doc.parsedContent?.data)) {
            console.log(` DEBUG Documento ${index} problematico:`, {
                name: doc.metadata?.name,
                type: doc.metadata?.type,
                hasContent: !!doc.parsedContent,
                chunkType: doc.parsedContent?.type,
                hasData: !!doc.parsedContent?.data
            });
        }
    }
    console.log("Location validation statistics:");
    const locationStats = {
        endpointsWithLocation: 0,
        endpointsWithoutLocation: 0,
        areasWithFloorInfo: 0,
        areasWithoutFloorInfo: 0
    };

    parsedDocs.forEach(doc => {
        if (doc.metadata.chunkType === 'detail') {
            if (doc.metadata.isFirstFloor !== undefined ||
                doc.metadata.isSecondFloor !== undefined) {
                locationStats.endpointsWithLocation++;
            } else {
                locationStats.endpointsWithoutLocation++;
            }
        }

        if (doc.metadata.chunkType === 'area') {
            if (doc.metadata.isFirstFloor !== undefined ||
                doc.metadata.isSecondFloor !== undefined) {
                locationStats.areasWithFloorInfo++;
            } else {
                locationStats.areasWithoutFloorInfo++;
            }
        }
    });

    console.log("Location validation:", locationStats);

    // Final statistics
    console.log(`Validation statistics:`, {
        totalDocuments: parsedDocs.length,
        endpointDocuments: endpointCount,
        installationDocuments: installationCount,
        areaDocuments: areaCount,
        documentsWithPartitions: documentsWithPartitions,
        documentsWithAreas: documentsWithAreas,
        validationResult: isValid ? 'VALID' : 'INVALID'
    });

    // Alerts if location information is missing
    if (endpointCount > 0 && documentsWithPartitions === 0 && documentsWithAreas === 0) {
        console.warn(" WARNING: No endpoint has location information (partitions or areas)");
    }
    // Check that all expected document types are present
    if (endpointCount === 0) {
        console.error("WARNING: No endpoint documents found");
        isValid = false;
    }

    if (installationCount === 0) {
        console.warn(" WARNING: No installation-config file found");
    }

    return isValid;
}



/**Fundamental function for:
Creates the mapping between areas and partitions
Manages both objects and UUID strings for partitions
Has robust validation
Required to resolve partition names */
export function buildAreaPartitionMaps(jsonContent: any): AreaPartitionMap[] {
    const maps: AreaPartitionMap[] = [];

    if (!jsonContent.areas || !Array.isArray(jsonContent.areas)) {
        console.warn("No areas found in JSON file");
        return maps;
    }

    for (const [index, area] of jsonContent.areas.entries()) {
        try {
            // RIGOROUS VALIDATION AREAendpointDoc
            if (!area || typeof area !== 'object') {
                console.warn(`Areas ${index} invalid:`, area);
                continue;
            }

            if (!area.uuid || !area.name) {
                console.warn(` Areas ${index} with UUID or name missing:`, {
                    uuid: area.uuid,
                    name: area.name
                });
                continue;
            }


            const areaMap: AreaPartitionMap = {
                areaUuid: area.uuid,
                areaName: area.name,
                partitions: []
            };


            // SECURE PARTITION PROCESSING
            if (Array.isArray(area.partitions)) {
                for (const [partIndex, partition] of area.partitions.entries()) {
                    if (!partition) {
                        console.warn(` Partizione ${partIndex} in area ${area.name} è null/undefined`);
                        continue;
                    }

                    // I Handle both objects and UUID strings
                    const partitionUuid = typeof partition === 'string' ? partition : partition.uuid;
                    const partitionName = typeof partition === 'string' ?
                        `Partition_${partition.substring(0, 8)}` : partition.name;

                    if (partitionUuid && partitionName) {
                        areaMap.partitions.push({
                            uuid: partitionUuid,
                            name: partitionName
                        });
                    } else {
                        console.warn(` Partition ${partIndex} in area ${area.name} with incomplete data`);
                    }
                }
            }

            maps.push(areaMap);
            console.log(` Mapped Area: ${area.name} (${areaMap.partitions.length} partitions)`);
        } catch (error) {
            console.error(` Error processing area ${index}:`, error);
            continue;
        }
    }
    console.log(`Area maps created: ${maps.length}`);
    return maps;
}


//Itera sugli endpoints e trova le aree attraverso partizioni condivise
function buildEndpointAreaRelations(
    jsonContent: any,
    areaPartitionMaps: AreaPartitionMap[]
): Map<string, EndpointAreaRelation> {
    const relations = new Map<string, EndpointAreaRelation>();

    console.log("  Costruzione relazioni endpoint-area...");

    // Validazione input
    if (!jsonContent?.areas || !Array.isArray(jsonContent.areas)) {
        console.warn("Nessuna area nel JSON per costruire relazioni");
        return relations;
    }

    if (!Array.isArray(areaPartitionMaps) || areaPartitionMaps.length === 0) {
        console.warn(" Nessuna mappa partizioni per costruire relazioni");
        return relations;
    }

    console.log(`Aree da processare: ${jsonContent.areas.length}, Mappe partizioni: ${areaPartitionMaps.length}`);
    console.log("Aree processate:", JSON.stringify(jsonContent.areas), "Nome partizoni", JSON.stringify(areaPartitionMaps));

    let totalEndpointsProcessed = 0;
    let totalRelationsCreated = 0;

    // for (const [areaIndex, area] of jsonContent.areas.entries()) {
    for (const [endpointIndex, endpoint] of jsonContent.endpoints.entries()) {

        try {
            totalEndpointsProcessed++;

            // Validazione endpoint
            if (!endpoint || !endpoint.uuid) {
                console.warn(`Endpoint ${endpointIndex} non valido o senza UUID`);
                continue;
            }

            // Se l'endpoint non ha partizioni, salta
            if (!Array.isArray(endpoint.partitions) || endpoint.partitions.length === 0) {
                continue;
            }

            const endpointName = endpoint.name || `Device_${endpoint.uuid.substring(0, 8)}`;

            // Per ogni area, controlla se condivide partizioni con questo endpoint
            for (const areaMap of areaPartitionMaps) {
                // Trova partizioni in comune tra endpoint e area
                const sharedPartitions = areaMap.partitions.filter(areaPartition =>
                    endpoint.partitions.includes(areaPartition.uuid)
                );

                // Se ci sono partizioni condivise, crea la relazione
                if (sharedPartitions.length > 0) {
                    const relation: EndpointAreaRelation = {
                        endpointUuid: endpoint.uuid,
                        endpointName: endpointName,
                        areaUuid: areaMap.areaUuid,
                        areaName: areaMap.areaName,
                        partitionUuids: sharedPartitions.map(p => p.uuid),
                        location: sharedPartitions.map(p => p.name)
                    };

                    // Verifica duplicati (un endpoint può essere in più aree)
                    if (relations.has(endpoint.uuid)) {
                        const existing = relations.get(endpoint.uuid);
                        console.log(`Endpoint ${endpoint.uuid} già mappato a ${existing?.areaName}, aggiungendo anche ${areaMap.areaName}`);
                        // In questo caso, potresti voler gestire relazioni multiple
                        // Per ora, manteniamo solo la prima relazione trovata
                    } else {
                        relations.set(endpoint.uuid, relation);
                        totalRelationsCreated++;

                        console.log(`Relazione creata: ${endpointName} -> ${areaMap.areaName} (${sharedPartitions.length} partizioni condivise)`);
                    }
                }
            }

        } catch (endpointError) {
            console.error(`Errore processing endpoint ${endpointIndex}:`, endpointError);
            continue;
        }
    }

    // REPORT FINALE
    console.log("\nREPORT RELAZIONI ENDPOINT-AREA:");
    console.log(`Endpoints processati: ${totalEndpointsProcessed}`);
    console.log(`Relazioni create: ${totalRelationsCreated}`);
    console.log(`Relazioni uniche nella mappa: ${relations.size}`);

    // DEBUG DETTAGLIATO
    if (relations.size > 0) {
        console.log("\nPrime 3 relazioni create:");
        let count = 0;
        for (const [uuid, relation] of relations.entries()) {
            if (count >= 3) break;
            console.log(`   ${count + 1}. ${relation.endpointName} -> ${relation.areaName}`);
            console.log(`      UUID: ${uuid}`);
            console.log(`      Partizioni condivise: ${relation.location.join(', ')}`);
            count++;
        }
    } else {
        console.warn("\nATTENZIONE: Nessuna relazione creata!");
        console.log("Debug: verifica che aree e endpoints condividano partizioni");

        // Debug delle partizioni per area
        console.log("Partizioni per area:");
        areaPartitionMaps.forEach(areas => {
            console.log(`   ${areas.areaName}: [${areas.partitions.map(p => p.name).join(', ')}]`);
        });

        // Debug delle partizioni per endpoint (primi 5)
        console.log("Partizioni per endpoint (primi 5):");
        jsonContent.endpoints.slice(0, 5).forEach((ep: any) => {
            if (ep.partitions?.length > 0) {
                console.log(`   ${ep.name}: [${ep.partitions.join(', ')}]`);
            }
        });
    }

    return relations;
    /* try {
         // Validazione area
         if (!area || typeof area !== 'object') {
             console.warn(`⚠️ Area ${areaIndex} non valida (non è un oggetto)`);
             continue;
         }
 
         if (!area.uuid || !area.name) {
             console.warn(`⚠️ Area ${areaIndex} senza UUID o nome:`, { uuid: area.uuid, name: area.name });
             continue;
         }
 
         // Validazione dispositivi area
         if (!Array.isArray(area.devices)) {
             console.log(`📭 Area ${area.name} senza array devices:`, typeof area.devices);
             continue;
         }
 
         console.log(`📍 Processing area [${areaIndex}]: ${area.name} (${area.uuid}), dispositivi: ${area.devices.length}`);
 
         // Trova mappa partizioni per questa area
         const areaMap = areaPartitionMaps.find(map => map.areaUuid === area.uuid);
         if (!areaMap) {
             console.warn(`⚠️ Mappa partizioni non trovata per area: ${area.name} (${area.uuid})`);
             console.log(" Mappe disponibili:", areaPartitionMaps.map(m => `${m.areaName} (${m.areaUuid})`));
             continue;
         }
 
         console.log(`   📋 Trovata mappa partizioni: ${areaMap.partitions.length} partizioni`);
 
         // Processa ogni dispositivo nell'area
         for (const [deviceIndex, device] of area.devices.entries()) {
             totalDevicesProcessed++;
 
             try {
                 // Validazione dispositivo
                 if (!device || typeof device !== 'object') {
                     console.warn(`   ⚠️ Dispositivo ${deviceIndex} in area ${area.name} non valido`);
                     continue;
                 }
 
                 if (!device.uuid) {
                     console.warn(`   ⚠️ Dispositivo ${deviceIndex} in area ${area.name} senza UUID`);
                     continue;
                 }
 
                 const deviceName = device.name || `Device_${device.uuid.substring(0, 8)}`;
 
                 // Crea relazione
                 const relation: EndpointAreaRelation = {
                     endpointUuid: device.uuid,
                     endpointName: deviceName,
                     areaUuid: area.uuid,
                     areaName: area.name,
                     partitionUuids: areaMap.partitions.map(p => p.uuid),
                     location: areaMap.partitions.map(p => p.name)
                 };
 
                 // Verifica duplicati
                 if (relations.has(device.uuid)) {
                     const existing = relations.get(device.uuid);
                     console.warn(`   ⚠️ UUID duplicato: ${device.uuid}`, {
                         existing: `${existing?.endpointName} -> ${existing?.areaName}`,
                         new: `${relation.endpointName} -> ${relation.areaName}`
                     });
                 }
 
                 relations.set(device.uuid, relation);
                 totalRelationsCreated++;
 
                 console.log(`    Relazione creata: ${deviceName} -> ${area.name} (${relation.location.length} partizioni)`);
 
             } catch (deviceError) {
                 console.error(`    Errore processing dispositivo ${deviceIndex} in area ${area.name}:`, deviceError);
                 continue;
             }
         }
 
     } catch (areaError) {
         console.error(`❌ Errore processing area ${areaIndex}:`, areaError);
         continue;
     }
 }
 
 // REPORT FINALE
 console.log("\n REPORT RELAZIONI ENDPOINT-AREA:");
 console.log(`Aree processate: ${jsonContent.areas.length}`);
 console.log(`\n Dispositivi totali: ${totalDevicesProcessed}`);
 console.log(`\n Relazioni create: ${totalRelationsCreated}`);
 console.log(`\n Relazioni uniche nella mappa: ${relations.size}`);
 
 // DEBUG DETTAGLIATO
 if (relations.size > 0) {
     console.log("\n Prime 3 relazioni create:");
     let count = 0;
     for (const [uuid, relation] of relations.entries()) {
         if (count >= 3) break;
         console.log(`   ${count + 1}. ${relation.endpointName} -> ${relation.areaName}`);
         console.log(`      UUID: ${uuid}`);
         console.log(`      Partizioni: ${relation.location.join(', ')}`);
         count++;
     }
 } else {
     console.warn("\n ATTENZIONE: Nessuna relazione creata!");
     console.log(" Debug struttura aree:");
     jsonContent.areas.forEach((area: any, i: number) => {
         console.log(`   Area ${i}: ${area.name}`, {
             uuid: area.uuid,
             hasDevices: Array.isArray(area.devices),
             devicesCount: area.devices?.length || 0,
             devicesSample: area.devices?.slice(0, 2).map((d: any) => ({
                 uuid: d.uuid,
                 name: d.name
             }))
         });
     });
 
     console.log(" Debug mappe partizioni:");
     areaPartitionMaps.forEach((map, i) => {
         console.log(`   Mappa ${i}: ${map.areaName}`, {
             uuid: map.areaUuid,
             partitionsCount: map.partitions.length,
             partitionsSample: map.partitions.slice(0, 2)
         });
     });
 }
 
 return relations;*/
}

function getFallbackDocument(error: any): { Documents: ExtendDocument[], partitionMap: Map<string, any> } {
    const fallbackUUID = 'fallback-' + Math.random().toString(36).substring(2, 9);
    const fallBack = {
        pageContent: JSON.stringify({
            error: "Failed to load document",
            message: error instanceof Error ? error.message : String(error),
            fallbackType: "empty_system"
        }),
        metadata: {
            source: 'fallback',
            loc: 'internal',
            type: 'installation-config' as const, //fallback
            isValid: false,
            timestamp: new Date().toISOString(),
            uuid: fallbackUUID,
            name: 'Fallback Document',
            category: -1,
            visualizationType: 'N/A',
            deviceType: 'other',
            // Campi opzionali con valori default
            categoryName: 'fallback',
            visualizationCategory: 'fallback',
            id: '0',
            partitions: [],
            location: [],
            areaNames: [],
            areaUuids: [],
            parametersCount: 0,
            defaultParameter: '',
            chunkStrategy: 'standard' as const,
            chunkType: 'fallback' as const,
            hasAreaInfo: true

        },
        readableText: "System temporarily unavailable. Please check the configuration and try again."
    };
    //"Document loading failed. Please check the configuration file."
    return {
        Documents: [fallBack],
        partitionMap: new Map()
    }
}

