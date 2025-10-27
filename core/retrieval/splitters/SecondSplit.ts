

import { Endpoint } from "src/config/RAG.js";
import { ChunkType, EndpointMetadata } from "../loaders/loadDocumentJSON3.js";




interface ProcessingMetadata extends EndpointMetadata {
    // Campi specifici per il processing
    chunkId?: string;
    parentChunkId?: string;
    structuralChunk?: boolean;
    sourceArray?: string;
    arrayIndex?: number;
    totalArrayItems?: number;
    subChunkIndex?: number;
    isSubChunk?: boolean;
    usbChunkIndex?: number;
    totalSubChunks?: number;
    splitField?: string;
    uniqueChunkId?: string;
    parentUuid?: string;
    parentName?: string;
    deviceCategory?: number;
}

export type Chunk = {
    pageContent: string;
    metadata: Record<string, any>;
};
//SECOND SPLIT IF CHUNK TOO BIG
/*export function splitLargeJsonObjectByArrayField(
    obj: any,
    maxChunkSize: number,
    metadata: EndpointMetadata,
    candidateFields: string[] = ["parameters", "readings", "signals", "items"]
): Chunk[] {

    console.log(`🔄 Splitting object of ${JSON.stringify(obj).length} chars`);
    console.log(`   Original metadata: ${JSON.stringify({
        chunkType: metadata.chunkType,
        source: metadata.source,
        uuid: metadata.uuid,
        name: metadata.name
    })}`);


    // PRESERVA TUTTI I METADATI CRITICI
    const criticalMetadata = {
        ...metadata,
        splitAttempted: true
    };
    try {
        // Gestione specifica per la struttura dei chunk installation-config
        const contentType = obj.contentType || metadata.chunkType;

        // Cerca campi array da splittare
        for (const field of candidateFields) {
            const array = obj.data[field];
            if (Array.isArray(array)) {
                const baseData = { ...obj.data || obj };
                delete baseData[field];

                const chunks: Chunk[] = [];

                for (const [index, item] of array.entries()) {
                    const partialObj = {
                        ...obj,
                        data: {
                            ...baseData,
                            [field]: [item]
                        }
                    };

                    const chunkMetadata = {
                        ...criticalMetadata,
                        subChunkIndex: index,
                        totalSubChunks: array.length,
                        splitField: field,
                        fullUuid: `${metadata.uuid || 'no-uuid'}-${field}-${index}`,
                        parametersCount: 1, // Per chunk di parametri singoli
                        isSubChunk: true
                    };

                    const str = JSON.stringify(partialObj);

                    if (str.length <= maxChunkSize) {
                        chunks.push({
                            pageContent: str,
                            metadata: {
                                ...metadata,
                                metadata: chunkMetadata
                            }

                        });


                        console.log(`Splitting field ${field} with ${array.length} items`);//Non sono sicuro vada qua

                    } else {
                        console.warn(`Element too large even on its own in '${field}':`, item);
                        // Se anche il singolo elemento è troppo grande, fallback
                        const fallback = {
                            ...obj,
                            data: {
                                ...baseData,
                                [field]: []
                            }
                        };
                        chunks.push({
                            pageContent: JSON.stringify(fallback),
                            metadata: {
                                ...metadata,
                                //splitField: field,
                                subChunkIndex: index,
                                warning: "Non-splitable element"
                            }
                        });
                    }
                }
                // returns after the first splittable field found
                return chunks;
            }
        }
        // Fallback if no array field is splittable 
        const str = JSON.stringify(obj);


        if (str.length > maxChunkSize * 2) {
            console.warn(`⚠️ Chunk too big (${str.length} chars), will be truncated.`);

            return [{
                pageContent: str.substring(0, maxChunkSize),
                metadata: {
                    ...metadata,
                    warning: "Truncated due to exceeding maximum size"
                }
            }];
        }

        //Otherwise everything comes back in one piece
        return [{
            pageContent: str,
            metadata
        }];
    } catch (error) {
        console.error("Error in splitLargeJsonObjectByArrayField function:", error);

        return [{
            pageContent: JSON.stringify(obj),
            metadata: {
                ...metadata,
                error: "Error parsing/splitting"
            }
        }];
    }
}*/



/**il problema corretto. Il RecursiveCharacterTextSplitter 
 * fallisce perché installation-config.json è un unico oggetto JSON compatto senza separatori naturali 
 * che il text splitter possa usare. 
 * 
 * Il nuovo approccio strutturale risolve il problema
Perché falliva prima:

RecursiveCharacterTextSplitter cerca separatori di testo (\n\n, ., spazi)
installation-config.json è un blob JSON compatto senza separatori naturali
Il sistema ciclava sempre sullo stesso oggetto di 37.005 caratteri

Come funziona la nuova soluzione:

Parsing della struttura JSON: Identifica arrays principali (area, partitions, endpoints, devices)
Split per elementi: Ogni elemento dell'array diventa un chunk separato
Fallback a proprietà: Se non trova arrays, divide per proprietà principali dell'oggetto
Validazione garantita: Controlla che ogni chunk sia ≤ chunkSize, altrimenti applica text splitting*/
export function splitLargeJsonObjectByArrayField(
    obj: any,
    maxChunkSize: number,
    metadata: EndpointMetadata,
    depth: number = 0 //perr evitare loop infiniti
): Chunk[] {

    let count = 0;

    console.log(` Splitting object of ${JSON.stringify(obj).length} chars`);

    console.log(`   Original metadata: ${JSON.stringify({
        chunkType: metadata.chunkType,
        source: metadata.source,
        uuid: metadata.uuid,
        name: metadata.name,
        isFirstFloor: metadata.isFirstFloor,
        isSecondFloor: metadata.isSecondFloor,
        floorLocation: metadata.floorLocation,
        partitionNames: metadata.partitionNames
    })}`);




    // Protezione ricorsione
    if (depth > 5) {
        console.warn("Max depth reached, truncating");
        return [{
            pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
            metadata: {
                ...metadata,
                warning: "Max depth reached - truncated",
                error: "Structural splitting failed"
            }
        }
        ];
    }


    try {

        // CORREZIONE: Calcola floorLocation se mancante o inconsistente
        /* const calculateFloorLocation = (first: any, second: any): string => {
             // Gestisci sia boolean che stringhe
             const isFirst = first === true || first === 'first' || first === 'true';
             const isSecond = second === true || second === 'second' || second === 'true';
 
             if (isFirst && isSecond) return 'both';
             if (isFirst) return 'first';
             if (isSecond) return 'second';
             return 'unknown';
         };*/

        const normalizelocationMetadata = (meta: EndpointMetadata): EndpointMetadata => {
            //Gestisci sia boolean che le stringhe
            const isFirst = meta.isFirstFloor === 'first' || meta.isFirstFloor === 'First' || meta.isFirstFloor === 'true';
            const isSecond = meta.isSecondFloor === 'second' || meta.isSecondFloor === 'Second' || meta.isSecondFloor === 'true';

            let floorLocation = meta.floorLocation;

            // Calcola floorLocation solo se non è già definito o è incoerente
            if (!floorLocation || floorLocation === 'unknown' || floorLocation === 'Unknown') {
                if (isFirst && isSecond) floorLocation = 'both';
                else if (isFirst) floorLocation = 'first';
                else if (isSecond) floorLocation = 'second';
                else floorLocation = 'unknown';
            }

            return {
                ...meta,
                isFirstFloor: isFirst ? 'first' : 'unknown',
                isSecondFloor: isSecond ? 'second' : 'unknown',
                floorLocation: floorLocation.toLowerCase()
            };


        };

        const normalizeMetadata = normalizelocationMetadata(metadata);


        /*        // CORREZIONE: Standardizza i flag floor
                const normalizeFloorFlag = (value: any, expectedValue: string): string => {
                    if (value === true || value === expectedValue || value === 'true') {
                        return expectedValue;
                    }
                    return 'Unknown';
                };
        
                // Aggiungi questo debug:
                /*  if (depth === 0) {
                      console.log("🔍 Top-level object keys:", Object.keys(obj));
                      if (obj.data) {
                          console.log("🔍 Data object keys:", Object.keys(obj.data));
                          console.log("🔍 Arrays in data:",
                              Object.keys(obj.data)
                                  .filter(k => Array.isArray(obj.data[k]))
                                  .map(k => `${k}: ${obj.data[k].length} items`)
                          );
                      }
                  }                                             /
        
                const locationMetadata = {
                    //isFirstFloor: metadata.isFirstFloor || "Unknown",
                    //isSecondFloor: metadata.isSecondFloor || "Unknown",
                    isFirstFloor: normalizeFloorFlag(metadata.isFirstFloor, 'First'),
                    isSecondFloor: normalizeFloorFlag(metadata.isSecondFloor, 'Second'),
                    floorLocation: metadata.floorLocation || calculateFloorLocation(metadata.isFirstFloor, metadata.isSecondFloor),
        
                    // (metadata.isFirstFloor === "first" ? "first" :
                    //  metadata.isSecondFloor === "second" ? "second" : "unknown"),
                    partitionNames: metadata.partitionNames || [],
                    location: metadata.location || [],
                    areaName: metadata.areaName,
                    areaUuid: metadata.areaUuid,
                    hasAreaInfo: metadata.hasAreaInfo
                }
        
        */

        // PRESERVA TUTTI I METADATI CRITICI
        const criticalMetadata: ProcessingMetadata = {
            /*  isFirstFloor: metadata.isFirstFloor || 'Unknown',
              isSecondFloor: metadata.isSecondFloor || 'Unknown',
              floorLocation: metadata.floorLocation || 'unknown',
              partitionNames: metadata.partitionNames || [],
              hasAreaInfo: metadata.hasAreaInfo || false,*/
            ...normalizeMetadata,
            // Assicurati che i campi critici per il filtering siano presenti
            visualizationTypes: metadata.visualizationTypes || obj.visualizationType || obj.visualizationTypes,
            categories: metadata.categories || obj.category !== undefined ? [obj.category] : undefined,
            category: metadata.category || obj.category,
            visualizationType: metadata.visualizationType || obj.visualizationType,

            //Flag for tracking
            splitAttempted: true,
            splitDepth: depth
        };


        // Gestione specifica per la struttura dei chunk installation-config
        const chunkType = obj.chunkType || metadata.chunkType;

        const topLevelArrays = ['area', 'partitions', 'endpoints', 'devices', 'parameters'];

        // CERCA ARRAY IN obj.data PRIMA CHE IN obj
        let targetObject = obj;
        if (obj.data && typeof obj.data === 'object') {
            targetObject = obj.data;
            console.log(" Switching to data object for array search");
        }

        // Cerca campi array da splittare - ora cerchiamo direttamente nell'oggetto, non in obj.data
        for (const field of topLevelArrays) {
            const array = targetObject[field]; // Cambiato da obj.data[field] a obj[field]

            if (Array.isArray(array) && array.length > 0) {
                const baseData = { ...targetObject };
                delete baseData[field];

                const chunks: Chunk[] = [];
                console.log(`Splitting field ${field} with ${array.length} items`);

                //Enumeration gerechic system

                //create parent chunk (only if is not present)

                //For parent chunks
                const parentChunk: Chunk = {
                    pageContent: JSON.stringify({
                        type: 'parent',
                        uuid: criticalMetadata.uuid,
                        name: criticalMetadata.name,
                        hasChildren: true,
                        childrenCount: array.length,
                        originalData: baseData //Mantain the context

                    }),
                    metadata: {
                        ...criticalMetadata, //====================================================> Da valutare se corretto!
                        chunkType: 'summary' as const,
                        chunkId: "0", //The father is always 0
                        isParent: true,
                        totalChildren: array.length,
                        hasSubChunks: true
                    }
                };




                /*    // Crea un chunk per ogni elemento dell'array
                    for (const [index, item] of array.entries()) {
                        const chunkObj = {
                            type: field.slice(0, -1), // "endpoints" -> "endpoint"
                            ...item,
                            // Mantieni il contesto dell'area se disponibile
                            areaContext: targetObject.areas ? targetObject.areas[0] : null
                        };*/



                //Create children
                for (const [index, item] of array.entries()) {

                    const chunkObj = {
                        type: field.slice(0, -1), // "endpoints" -> "endpoint"
                        ...item,
                        parentContext: {
                            uuid: criticalMetadata.uuid,
                            name: criticalMetadata.name,
                            type: criticalMetadata.type
                        }

                    };


                    const str = JSON.stringify(chunkObj);



                    //DEBUGG
                    console.log(`👶 Creating child ${index} for field ${field}`);
                    console.log(`📋 Parent metadata:`, {
                        parentViz: criticalMetadata.visualizationTypes,
                        parentCat: criticalMetadata.categories,
                        parentVizType: criticalMetadata.visualizationType,
                        parentCategory: criticalMetadata.category
                    });

                    console.log(`📋 Item metadata:`, {
                        itemViz: item.visualizationTypes || item.visualizationType,
                        itemCat: item.categories || item.category
                    });




                    const processingMetadata: ProcessingMetadata = {
                        ...criticalMetadata,
                        chunkType: 'detail' as const,
                        chunkId: `0.${index + 1}`, // Sistema gerarchico 0.1, 0.2, 0.3
                        parentChunkId: "0", //Explicit reference to father
                        structuralChunk: true,
                        sourceArray: field,
                        arrayIndex: index,
                        totalArrayItems: array.length,
                        subChunkIndex: count++,
                        uuid: item.uuid || item.id || `${field}-${index}`,
                        name: item.name || `${field} ${index + 1}`,
                        isSubChunk: true,
                        usbChunkIndex: index,  //Indice unico
                        totalSubChunks: array.length, // Totale chunk
                        splitField: field, // Campo splittato
                        uniqueChunkId: `${metadata.uuid}-params-${index}-${field}`, // ID univoco

                        //Link to device father
                        parentUuid: criticalMetadata.uuid,
                        parentName: criticalMetadata.name,
                        deviceCategory: criticalMetadata.category,

                    };

                    //DEBUG
                    console.log(`✅ Child ${index} final metadata:`, {
                        viz: processingMetadata.visualizationTypes,
                        cat: processingMetadata.categories
                    });



                    if (str.length <= maxChunkSize) {
                        //Creo il chunk finale
                        /*chunks.push({
                            pageContent: str,
                            metadata: chunkMetadata // Cambiato: metadata diretto, non annidato
                        });*/
                        const finalChunks: Chunk = {
                            pageContent: str,
                            metadata: processingMetadata
                        };

                        console.log(`🎯 FINAL CHUNK METADATA:`, {
                            name: metadata.name,
                            visualizationTypes: metadata.visualizationTypes,
                            categories: metadata.categories,
                            visualizationType: metadata.visualizationType,
                            category: metadata.category
                        });


                        chunks.push(finalChunks);
                    } else {
                        //console.warn(`Element too large even on its own in '${field}':`, item); ------> stampa una miriade di roba

                        // Se anche il singolo elemento è troppo grande, fallback
                        // Ricorsione per elementi troppo grandi
                        console.log(`🔁 Recursing into oversized ${field} item ${index}`);
                        const subChunks = splitLargeJsonObjectByArrayField(
                            chunkObj,
                            maxChunkSize,
                            processingMetadata as EndpointMetadata,// chunkMetadata, //Dovrei modificarlo in chunckMetadata ma mi da errori di incompatibilità
                            depth + 1
                        );
                        chunks.push(...subChunks);

                    }
                }

                //  DEBUG: Verifica che i metadata di location siano preservati
                console.log(`   Split completed for ${field}. Chunks created: ${chunks.length}`);
                if (chunks.length > 0) {
                    const firstChunk = chunks[0].metadata;
                    console.log(`   First chunk location metadata:`, {
                        isFirstFloor: firstChunk.isFirstFloor,
                        isSecondFloor: firstChunk.isSecondFloor,
                        floorLocation: firstChunk.floorLocation,
                        partitionNames: firstChunk.partitionNames
                    });

                    // VALIDAZIONE: Assicurati che floorLocation non sia undefined
                    if (!firstChunk.floorLocation || firstChunk.floorLocation === 'undefined') {
                        console.warn(`⚠️ FloorLocation is ${firstChunk.floorLocation} for chunk ${firstChunk.name}`);
                    }
                }


                //Return father + children
                console.log(`✅ Created hierarchical chunks: 1 parent + ${chunks.length} children`);
                return [parentChunk, ...chunks];



                // returns after the first splittable field found
                // return chunks;
            }
        }
        // FALLBACK: Se non trova array, splitta per proprietà principali
        console.log(" No arrays found, attempting property-based split");
        return splitByProperties(obj, maxChunkSize, criticalMetadata, depth);

    } catch (error) {
        console.error(" Error in splitLargeJsonObjectByArrayField:", error);

        let errorMessages = "Uknown error";
        if (error instanceof Error) errorMessages = error.message;

        return [{

            pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
            metadata: {
                ...metadata,
                error: `Splitting error: ${errorMessages}`
            }
        }];
    }

}

// FUNZIONE DI FALLBACK PER SPLITTARE PER PROPRIETÀ
function splitByProperties(obj: any, maxChunkSize: number, metadata: EndpointMetadata, depth: number): Chunk[] {
    const chunks: Chunk[] = [];
    const properties = ['area', 'endpoints', 'metadata', 'configurations'];

    for (const prop of properties) {
        if (obj[prop] && typeof obj[prop] === 'object') {
            const propChunk = {
                type: prop,
                [prop]: obj[prop]
            };

            const str = JSON.stringify(propChunk);
            if (str.length <= maxChunkSize) {
                chunks.push({
                    pageContent: str,
                    metadata: {
                        ...metadata,
                        chunkType: 'partial' as ChunkType,
                        splitProperty: prop,
                        floorLocation: metadata.floorLocation || 'Unknown'
                    }
                }
                );
            }
        }
    }

    if (chunks.length > 0) {
        console.log(`✅ Fallback split created ${chunks.length} chunks with location metadata`);
        return chunks;
    }

    return [{
        pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
        metadata: {
            ...metadata,
            warning: "Fallback truncation",
            originalSize: JSON.stringify(obj).length,
            floorLocation: metadata.floorLocation || 'unknown'
        }
    }
    ];
}