"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.splitLargeJsonObjectByArrayField = splitLargeJsonObjectByArrayField;
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
function splitLargeJsonObjectByArrayField(obj, //rappresenta il dispositivo/area completo, dati originali
maxChunkSize, 
// metadata: EndpointMetadata,
chunk, depth //perr evitare loop infiniti
) {
    var _a;
    if (depth === void 0) { depth = 0; }
    var count = 0;
    var metadata = chunk.metadata;
    console.log(" Splitting object of ".concat(JSON.stringify(obj).length, " chars"));
    console.log("   Original metadata: ".concat(JSON.stringify({
        chunkType: metadata.maxChunkSize.metadataType,
        source: metadata.source,
        uuid: metadata.metadata.uuid,
        name: metadata.metadata.name,
        cateogy: metadata.metadata.category,
        isFirstFloor: metadata.metadata.isFirstFloor,
        isSecondFloor: metadata.metadata.isSecondFloor,
        floorLocation: metadata.metadata.floorLocation,
        partitionNames: metadata.metadata.partitionNames,
        visualizationType: metadata.metadata.visualizationType,
        category: metadata.metadata.category
    })));
    console.log("\uD83D\uDD0D DEBUG splitLargeJsonObjectByArrayField:");
    console.log("   - Input object keys: ".concat(Object.keys(obj).join(', ')));
    console.log("   - Metadata: ".concat(JSON.stringify({
        name: metadata.name,
        chunkType: metadata.chunkType,
        source: metadata.source
    })));
    // Verifica se sta ricevendo la struttura corretta
    if (obj.data) {
        console.log("   - Data object keys: ".concat(Object.keys(obj.data).join(', ')));
    }
    if (obj.endpoints) {
        console.log("   - Endpoints count: ".concat(obj.endpoints.length));
    }
    // Protezione ricorsione
    if (depth > 5) {
        console.warn("Max depth reached, truncating");
        return [{
                pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
                metadata: __assign(__assign({}, metadata), { warning: "Max depth reached - truncated", error: "Structural splitting failed" })
            }
        ];
    }
    try {
        var normalizeMetadata = normalizelocationMetadata(metadata);
        // PRESERVA TUTTI I METADATI CRITICI
        var criticalMetadata = __assign(__assign({}, normalizeMetadata), { splitAttempted: true, visualizationType: metadata.visualizationType, category: metadata.category });
        // Gestione specifica per la struttura dei chunk installation-config
        // const chunkType = obj.chunkType || metadata.chunkType;
        // CERCA ARRAY IN obj.data PRIMA CHE IN obj
        var targetObject_1 = obj; //targetObject rappresenta i dati specifici, lo usiamo per lo splitting degli array
        if (obj.parameterData && typeof obj.parameterData === 'object') {
            console.log("   - Switching to parameterData for array search");
            targetObject_1 = obj.parameterData;
        }
        else if (obj.data && typeof obj.data === 'object') {
            targetObject_1 = obj.data;
            console.log(" Switching to data object for array search");
        }
        console.log("   - Final target object ".concat(Object.keys(targetObject_1).join(', ')));
        var topLevelArrays = ['area', 'partitions', 'endpoints', 'devices', 'parameters'];
        //DEBUG for visula what arrays is found
        console.log("   -Checking for arrays in targetObject");
        topLevelArrays.forEach(function (field) {
            if (Array.isArray(targetObject_1[field])) {
                console.log("  FOUND: ".concat(field, " with ").concat(targetObject_1[field].length));
            }
        });
        //Versione con Sequence number per collegare children a father
        var splitSessionId = "split-".concat(Date.now(), "-").concat(Math.random().toString(36).substring(2, 9));
        var totalChunks = calculateTotalChunks(targetObject_1, topLevelArrays); //Function to implement
        //Create a father chunk
        var parentChunk = createParentChunk(obj, criticalMetadata, splitSessionId, totalChunks);
        var chunks = [parentChunk];
        //create chunk father (header)
        var chunkIndex = 1; //Inizia da 1 (0) il padre
        // Cerca campi array da splittare - ora cerchiamo direttamente nell'oggetto, non in obj.data
        for (var _i = 0, topLevelArrays_1 = topLevelArrays; _i < topLevelArrays_1.length; _i++) {
            var field = topLevelArrays_1[_i];
            var array = targetObject_1[field]; // Cambiato da obj.data[field] a obj[field]
            if (Array.isArray(array) && array.length > 0) {
                // const baseData = { ...targetObject };
                console.log("Splitting field ".concat(field, " with ").concat(array.length, " items"));
                for (var _b = 0, _c = array.entries(); _b < _c.length; _b++) {
                    var _d = _c[_b], index = _d[0], item = _d[1];
                    var chunkMetadata = __assign(__assign({}, criticalMetadata), { chunkType: 'detail', 
                        //System SequenceNumb like IP
                        sequenceNumberSystem: {
                            sessionId: splitSessionId,
                            chunkId: chunkIndex,
                            totalChunks: totalChunks,
                            parentChunkId: 0, //0 = chunk padre
                            isSequneceNumbChunk: true,
                            role: chunkIndex === 0 ? 'parent_device' : 'device_parameter',
                            relationship: chunkIndex === 0 ? 'contains_parameters' : "parameter_of_device_".concat(metadata.uuid)
                        }, chunkId: "0.".concat(index + 1), parentChunkId: "0", structuralChunk: true, sourceArray: field, arrayIndex: index, totalArrayItems: array.length, subChunkIndex: count++, uuid: item.uuid || item.id || "".concat(field, "-").concat(index), name: item.name || "".concat(field, " ").concat(index + 1), 
                        /*
                        
                        / UUID unico per ogni chunk figlio (evita deduplicazione)
                        uuid: `${criticalMetadata.uuid}-param-${index}`,
  
                         // ⚠️ NOME del device padre (per query testuali)*/
                        fatherName: criticalMetadata.name, isSubChunk: true, usbChunkIndex: index, totalSubChunks: array.length, splitField: field, uniqueChunkId: "".concat(metadata.uuid, "-params-").concat(index, "-").concat(field), 
                        //Link to device father, hierarchic relation
                        parentUuid: criticalMetadata.uuid, parentName: criticalMetadata.name, deviceCategory: criticalMetadata.category, visualizationType: criticalMetadata.visualizationType, hierarchicalRole: chunkIndex === 0 ? 'device_parent' : 'parameter_child', parentDeviceName: metadata.name, parentDeviceCategory: metadata.category, 
                        // Metadati specifici del parametro
                        isParameterChunk: true, parameterIndex: index, parameterName: item.name, parameterDataType: item.dataType, 
                        // ⚠️ Eredita tutti i flag di query del parent
                        hasControlParams: criticalMetadata.hasControlParams, hasMeasurementParams: criticalMetadata.hasMeasurementParams, hasConfigParams: criticalMetadata.hasConfigParams, hasAutomationParams: criticalMetadata.hasAutomationParams, 
                        // Location metadata ereditata
                        isFirstFloor: criticalMetadata.isFirstFloor, isSecondFloor: criticalMetadata.isSecondFloor, floorLocation: criticalMetadata.floorLocation, partitionNames: criticalMetadata.partitionNames });
                    var childChunk = createChildChunk(item, chunkMetadata, field, index);
                    if (childChunk.pageContent.length <= maxChunkSize) {
                        chunks.push(childChunk);
                    }
                    else {
                        console.log("\uD83D\uDD01 Recursing into oversized ".concat(field, " item ").concat(index));
                        var subChunk = splitLargeJsonObjectByArrayField(JSON.parse(childChunk.pageContent), maxChunkSize, 
                        //Creo oggetto Chunk che unisce anche oggetto EndpointMetadata
                        {
                            pageContent: childChunk.pageContent,
                            metadata: chunkMetadata
                        }, depth + 1);
                        chunks.push.apply(chunks, subChunk);
                    }
                    chunkIndex++;
                    //      STAMPA CHUNKS CHILD
                    console.log(" This is a child ".concat(JSON.stringify(childChunk), "- ").concat((_a = metadata.ackMetadata) === null || _a === void 0 ? void 0 : _a.sessionId, "  of ---> ").concat(JSON.stringify(parentChunk), " with \n                    ").concat(JSON.stringify(chunkIndex)));
                }
                console.log("   Split completed for ".concat(field, ". Chunks created: ").concat(chunks.length));
            }
            /*   const str = JSON.stringify(chunkObj);
 
               if (str.length <= maxChunkSize) {
                   chunks.push({
                       pageContent: str,
                       metadata: chunkMetadata // Cambiato: metadata diretto, non annidato
                   });
               } else {
                   //console.warn(`Element too large even on its own in '${field}':`, item); ------> stampa una miriade di roba
 
                   // Se anche il singolo elemento è troppo grande, fallback
                   // Ricorsione per elementi troppo grandi
                   console.log(`🔁 Recursing into oversized ${field} item ${index}`);
                   const subChunks = splitLargeJsonObjectByArrayField(
                       chunkObj,
                       maxChunkSize,
                       chunkMetadata, //Dovrei modificarlo in chunckMetadata ma mi da errori di incompatibilità
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
             }*/
        }
        if (chunks.length > 0) {
            return chunks;
        }
        // FALLBACK: Se non trova array, splitta per proprietà principali
        console.log(" No arrays found, attempting property-based split");
        return splitByProperties(obj, maxChunkSize, criticalMetadata, depth);
    }
    catch (error) {
        console.error(" Error in splitLargeJsonObjectByArrayField:", error);
        var errorMessages = "Uknown error";
        if (error instanceof Error)
            errorMessages = error.message;
        return [{
                pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
                metadata: __assign(__assign({}, metadata), { error: "Splitting error: ".concat(errorMessages) })
            }];
    }
}
function createParentChunk(originalObj, baseMetadata, sessionId, totalChunks) {
    var parentContent = {
        type: 'parent',
        ackHeader: {
            sessionId: sessionId,
            chunkId: 0, // ID 0 = chunk padre
            totalChunks: totalChunks,
            message: "This device has ".concat(totalChunks - 1, " parameter chunks. Refer to chunks 1-").concat(totalChunks - 1, " for details.")
        },
        deviceInfo: {
            name: baseMetadata.name,
            uuid: baseMetadata.uuid,
            category: baseMetadata.category
        }
    };
    return {
        pageContent: JSON.stringify(parentContent),
        metadata: __assign(__assign({}, baseMetadata), { chunkType: 'summary', sequenceNumberSystem: {
                sessionId: sessionId,
                chunkId: 0,
                totalChunks: totalChunks,
                isParent: true
            } })
    };
}
function createChildChunk(item, metadata, field, index) {
    var chunkObj = {
        type: 'parameter',
        ackInfo: {
            sessionId: metadata.sequenceNumberSystem.sessionId,
            chunkId: metadata.sequenceNumberSystem.chunkId,
            parentChunkId: 0,
            parameterIndex: index
        },
        parameterData: item
    };
    //Preseva che critical metadatas of parent
    var preservedMetadata = __assign(__assign({}, metadata), { visualizationType: metadata.visualizationType, category: metadata.category, categoryName: metadata.categoryName, deviceType: metadata.deviceType, name: metadata.name, uuid: metadata.uuid, 
        // Location metadata
        isFirstFloor: metadata.isFirstFloor, isSecondFloor: metadata.isSecondFloor, floorLocation: metadata.floorLocation, partitionNames: metadata.partitionNames, 
        // Flags di query
        hasControlParams: metadata.hasControlParams, hasMeasurementParams: metadata.hasMeasurementParams, hasConfigParams: metadata.hasConfigParams });
    return {
        pageContent: JSON.stringify(chunkObj),
        metadata: preservedMetadata
    };
}
function calculateTotalChunks(targetObject, fields) {
    var total = 1; // Sempre almeno il chunk padre
    for (var _i = 0, fields_1 = fields; _i < fields_1.length; _i++) {
        var field = fields_1[_i];
        var array = targetObject[field];
        if (Array.isArray(array)) {
            total += array.length;
        }
    }
    return total;
}
// FUNZIONE DI FALLBACK PER SPLITTARE PER PROPRIETÀ
function splitByProperties(obj, maxChunkSize, metadata, depth) {
    var _a;
    var chunks = [];
    var properties = ['area', 'endpoints', 'metadata', 'configurations'];
    for (var _i = 0, properties_1 = properties; _i < properties_1.length; _i++) {
        var prop = properties_1[_i];
        if (obj[prop] && typeof obj[prop] === 'object') {
            var propChunk = (_a = {
                    type: prop
                },
                _a[prop] = obj[prop],
                _a);
            var str = JSON.stringify(propChunk);
            if (str.length <= maxChunkSize) {
                chunks.push({
                    pageContent: str,
                    metadata: __assign(__assign({}, metadata), { chunkType: 'partial', splitProperty: prop, floorLocation: metadata.floorLocation || 'Unknown' })
                });
            }
        }
    }
    if (chunks.length > 0) {
        console.log("\u2705 Fallback split created ".concat(chunks.length, " chunks with location metadata"));
        return chunks;
    }
    return [{
            pageContent: JSON.stringify(obj).substring(0, maxChunkSize),
            metadata: __assign(__assign({}, metadata), { warning: "Fallback truncation", originalSize: JSON.stringify(obj).length, floorLocation: metadata.floorLocation || 'unknown' })
        }
    ];
}
function normalizelocationMetadata(meta) {
    //Gestisci sia boolean che le stringhe
    var isFirst = meta.isFirstFloor === 'first' || meta.isFirstFloor === 'First' || meta.isFirstFloor === 'true';
    var isSecond = meta.isSecondFloor === 'second' || meta.isSecondFloor === 'Second' || meta.isSecondFloor === 'true';
    var floorLocation = meta.floorLocation;
    // Calcola floorLocation solo se non è già definito o è incoerente
    if (!floorLocation || floorLocation === 'unknown' || floorLocation === undefined) {
        if (isFirst && isSecond)
            floorLocation = 'both';
        else if (isFirst)
            floorLocation = 'first';
        else if (isSecond)
            floorLocation = 'second';
        else
            floorLocation = 'unknown';
    }
    return __assign(__assign({}, meta), { isFirstFloor: isFirst ? 'first' : 'unknown', isSecondFloor: isSecond ? 'second' : 'unknown', floorLocation: floorLocation.toLowerCase() });
}
;
