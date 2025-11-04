
import { ExtendDocument, targetFile, filePath } from "../../../config/RAGNLP.js";
import { promises as fs } from "fs";
import { Document } from "langchain/document";//Document non è un array e non ha un metodo push
import { buildGlobalPartitionMap } from "./buildGlobalPartitionsMap.js";

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

// INTERFACCE PER MIGLIORARE LA MAPPATURA
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

export interface LoadDocumentResult {
    documents: ExtendDocument[],
    partitionMap: Map<string, string>
}



export async function loadDocumentsJSON(): Promise<LoadDocumentResult> {
    const processedUUIDs = new Set<string>();
    let rawContent: string;
    const documents: ExtendDocument[] = [];

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

    const mainDocument = new Document({
        pageContent: JSON.stringify(installationContent),
        metadata: {
            source: targetFile,
            loc: filePath,
            type: 'intallation-config',
            isValid: true,
            timestamp: new Date().toISOString(),
            name: jsonContent.metadata?.name,
            chunkType: 'summary',

            installationName: jsonContent.metadata?.name || 'installation-config',
            revision: jsonContent.metadata?.revision,
            deviceType: 'installation',
            totalEndpoints: jsonContent.endpoints?.length || 0,
            totalAreas: jsonContent.areas?.length || 0,
            hasPartitions: globalPartitionMap.size > 0,
            hasAreaInfo: hasValidAreas,
            major: jsonContent.metadata?.major,
            minor: jsonContent.metadata?.minor,


        }
    }) as unknown as ExtendDocument;
    documents.push(mainDocument);

    console.log("Single raw document created for two-stage chunking");
    return {
        documents,
        partitionMap: globalPartitionMap
    }
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

    // DETAILED DEBUG
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

}

function getFallbackDocument(error: any): LoadDocumentResult {
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
        documents: [fallBack],
        partitionMap: new Map()
    }
}

