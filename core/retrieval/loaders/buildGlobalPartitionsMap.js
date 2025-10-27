"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.buildGlobalPartitionMap = buildGlobalPartitionMap;
function buildGlobalPartitionMap(jsonContent, documents) {
    var map = new Map();
    /*console.log("🗺️ DEBUG: Inizio costruzione mappa globale partizioni", {
        hasJsonContent: !!jsonContent,
        hasAreas: !!jsonContent?.areas,
        hasPartitions: !!jsonContent?.partitions,
        areasCount: jsonContent?.areas?.length || 0,
        partitionsCount: jsonContent?.partitions?.length || 0,
        hasDocuments: !!documents,
        documentsCount: documents?.length || 0
    });*/
    // ========================================
    // STRATEGY 1: PARTITIONS FROM AREAS (ORIGINAL JSON)
    // ========================================
    console.log(" Building partition map from JSON structure...");
    // PRIMA STRATEGIA: Partizioni dalle aree (la più importante)
    if ((jsonContent === null || jsonContent === void 0 ? void 0 : jsonContent.areas) && Array.isArray(jsonContent.areas)) {
        console.log(" Processing ".concat(jsonContent.areas.length, " areas for partitions..."));
        for (var _i = 0, _a = jsonContent.areas; _i < _a.length; _i++) {
            var area = _a[_i];
            if (!area || !area.partitions || !Array.isArray(area.partitions))
                continue;
            for (var _b = 0, _c = area.partitions; _b < _c.length; _b++) {
                var partition = _c[_b];
                try {
                    if (typeof partition === 'object' && partition.uuid && partition.name) {
                        // Questo è il caso IDEALE: oggetto con UUID e nome
                        map.set(partition.uuid, partition.name);
                        console.log(" Mapped: ".concat(partition.uuid, " -> ").concat(partition.name));
                    }
                    else if (typeof partition === 'string' && partition) {
                        // Questo è un problema: UUID senza nome, cerchiamo di risolverlo
                        var partitionName = findPartitionNameInArea(partition, area);
                        map.set(partition, partitionName);
                        console.log(" String partition mapped: ".concat(partition, " -> ").concat(partitionName));
                    }
                }
                catch (error) {
                    console.error("Error processing partition:", error);
                }
            }
        }
    }
    // SECONDA STRATEGIA: Cerca partizioni mancanti negli endpoint
    if ((jsonContent === null || jsonContent === void 0 ? void 0 : jsonContent.endpoints) && Array.isArray(jsonContent.endpoints)) {
        console.log(" Scanning endpoints for missing partition mappings...");
        for (var _d = 0, _e = jsonContent.endpoints; _d < _e.length; _d++) {
            var endpoint = _e[_d];
            if (!endpoint.partitions || !Array.isArray(endpoint.partitions))
                continue;
            for (var _f = 0, _g = endpoint.partitions; _f < _g.length; _f++) {
                var partitionUuid = _g[_f];
                if (typeof partitionUuid === 'string' && !map.has(partitionUuid)) {
                    // Partizione non trovata, cerchiamo di dedurne il nome
                    var partitionName = findPartitionNameFromContext(partitionUuid, jsonContent);
                    map.set(partitionUuid, partitionName);
                    console.log(" Deduced from context: ".concat(partitionUuid, " -> ").concat(partitionName));
                }
            }
        }
    }
    console.log(" Partition map completed: ".concat(map.size, " mappings"));
    // Debug: mostra tutte le mappings
    console.log(" Final partition mappings:");
    for (var _h = 0, _j = map.entries(); _h < _j.length; _h++) {
        var _k = _j[_h], uuid = _k[0], name_1 = _k[1];
        console.log("   ".concat(uuid, " -> ").concat(name_1));
    }
    return map;
}
// Helper function per trovare nomi partizioni nel contesto
function findPartitionNameFromContext(uuid, jsonContent) {
    // Cerca nelle aree
    if (jsonContent.areas) {
        for (var _i = 0, _a = jsonContent.areas; _i < _a.length; _i++) {
            var area = _a[_i];
            if (area.partitions) {
                for (var _b = 0, _c = area.partitions; _b < _c.length; _b++) {
                    var partition = _c[_b];
                    if (typeof partition === 'object' && partition.uuid === uuid && partition.name) {
                        return partition.name;
                    }
                }
            }
        }
    }
    // Fallback: usa l'UUID abbreviato
    return "Partition_".concat(uuid.substring(0, 8));
}
function findPartitionNameInArea(uuid, area) {
    // Se l'area ha un nome, usalo per creare un nome migliore
    if (area.name) {
        return "".concat(area.name, "_Partition");
    }
    return "Partition_".concat(uuid.substring(0, 8));
}
/*   if (jsonContent?.areas && Array.isArray(jsonContent.areas)) {
       console.log(` Processing ${jsonContent.areas.length} aree per partizioni...`);

       for (const [areaIndex, areas] of jsonContent.areas.entries()) {
           try {
               if (!areas || typeof areas !== 'object') {
                   console.warn(`⚠️ Area ${areaIndex} non valida:`, typeof areas);
                   continue;
               }

               if (!areas.uuid || !areas.name) {
                   console.warn(`⚠️ Area ${areaIndex} con UUID o nome mancante:`, {
                       uuid: areas.uuid,
                       name: areas.name
                   });
                   continue;
               }

               // console.log(`📍 Processing area: ${area.name} (${area.uuid})`);

               if (Array.isArray(areas.partitions) && areas.partitions.length > 0) {
                   console.log(`  Found ${areas.partitions.length} partitons in areas ${areas.name}`);

                   for (const [partIndex, partition] of areas.partitions.entries()) {
                       try {
                           if (!partition) {
                               console.warn(`    Partizione ${partIndex} in area ${areas.name} è null/undefined`);
                               continue;
                           }

                           let partitionUuid: string;
                           let partitionName: string;

                           if (typeof partition === 'string') {
                               partitionUuid = partition;
                               partitionName = `Partition_${partition.substring(0, 8)}`;
                               console.log(`     Partizione stringa: ${partitionUuid} -> ${partitionName}`);
                           } else if (typeof partition === 'object') {
                               partitionUuid = partition.uuid;
                               partitionName = partition.name;

                               if (!partitionUuid || !partitionName) {
                                   console.warn(`     Partizione oggetto ${partIndex} in area ${areas.name} con dati incompleti:`, {
                                       uuid: partitionUuid,
                                       name: partitionName
                                   });
                                   continue;
                               }
                               console.log(`     Partizione oggetto: ${partitionUuid} -> ${partitionName}`);
                           } else {
                               console.warn(`     Partizione ${partIndex} in area ${areas.name} ha tipo non supportato:`, typeof partition);
                               continue;
                           }

                           if (map.has(partitionUuid)) {
                               const existingName = map.get(partitionUuid);
                               if (existingName !== partitionName) {
                                   console.warn(`     UUID partizione duplicato con nomi diversi: ${partitionUuid}`, {
                                       existing: existingName,
                                       new: partitionName
                                   });
                               }
                           } else {
                               map.set(partitionUuid, partitionName);
                               console.log(`     Mappata: ${partitionUuid} -> ${partitionName}`);
                           }

                       } catch (partitionError) {
                           console.error(`     Errore processing partizione ${partIndex} in area ${areas.name}:`, partitionError);
                           continue;
                       }
                   }
               } else {
                   console.log(`   Area ${areas.name} senza partizioni`);
               }

           } catch (areaError) {
               console.error(` Errore processing area ${areaIndex}:`, areaError);
               continue;
           }
       }
   } else {
       console.warn(" Nessuna area trovata o array aree non valido");
   }

   // ========================================
   // STRATEGY 2: GLOBAL PARTITIONS (ORIGINAL JSON)
   // ========================================
   if (jsonContent?.partitions && Array.isArray(jsonContent.partitions)) {
       console.log(` Processing ${jsonContent.partitions.length} partizioni globali...`);

       for (const [index, partition] of jsonContent.partitions.entries()) {
           try {
               if (!partition || typeof partition !== 'object') {
                   console.warn(` Global partition ${index} invalid:`, typeof partition);
                   continue;
               }

               if (!partition.uuid || !partition.name) {
                   console.warn(`Global partition ${index} with incomplete data:`, {
                       uuid: partition.uuid,
                       name: partition.name
                   });
                   continue;
               }

               if (map.has(partition.uuid)) {
                   const existingName = map.get(partition.uuid);
                   console.log(` Updating partition from globals: ${partition.uuid}`, {
                       from: existingName,
                       to: partition.name
                   });
               }

               map.set(partition.uuid, partition.name);
               console.log(` Partizione globale mappata: ${partition.uuid} -> ${partition.name}`);

           } catch (globalPartitionError) {
               console.error(` Errore processing partizione globale ${index}:`, globalPartitionError);
               continue;
           }
       }
   } else {
       console.log("📭 Nessuna partizione globale trovata");
   }

   // ========================================
   // STRATEGY 3: PARTITIONS FROM DOCUMENTS (PAGE CONTENT)
   // ========================================
   if (documents && Array.isArray(documents)) {
       console.log(` Processing ${documents.length} documenti per partizioni aggiuntive...`);

       for (const [docIndex, doc] of documents.entries()) {
           try {
               if (!doc?.pageContent) continue;

               let parsedContent: any;
               try {
                   parsedContent = JSON.parse(doc.pageContent);
               } catch (parseError) {
                   console.warn(` Documento ${docIndex} non parsabile:`, parseError);
                   continue;
               }

               // Extract partitions from pageContent based on document structure
               const partitionsToProcess: Array<{ uuid: string, name: string }> = [];

               // Case 1: Area document
               if (parsedContent.type === 'area' && parsedContent.data?.partitions) {
                   parsedContent.data.partitions.forEach((p: any) => {
                       if (typeof p === 'object' && p.uuid) {
                           partitionsToProcess.push({
                               uuid: p.uuid,
                               name: p.name || `Partition_${p.uuid.substring(0, 8)}`
                           });
                       } else if (typeof p === 'string') {
                           partitionsToProcess.push({
                               uuid: p,
                               name: `Partition_${p.substring(0, 8)}`
                           });
                       }
                   });
               }

               // Case 2: Endpoint document with partitions
               else if (parsedContent.type === 'endpoint' && parsedContent.data?.partitions) {
                   const partitionNames = parsedContent.data.partitionNames || [];
                   parsedContent.data.partitions.forEach((uuid: string, index: number) => {
                       const name = partitionNames[index] || `Partition_${uuid.substring(0, 8)}`;
                       partitionsToProcess.push({ uuid, name });
                   });
               }

               // Case 3: Installation-config document
               else if (parsedContent.type === 'installation-config' && parsedContent.overview?.area) {
                   parsedContent.overview.area.forEach((area: any) => {
                       if (area.partitions) {
                           area.partitions.forEach((p: any) => {
                               if (typeof p === 'object' && p.uuid) {
                                   partitionsToProcess.push({
                                       uuid: p.uuid,
                                       name: p.name || `Partition_${p.uuid.substring(0, 8)}`
                                   });
                               }
                           });
                       }
                   });
               }

               // Aggiungi tutte le partizioni trovate alla mappa
               partitionsToProcess.forEach(({ uuid, name }) => {
                   if (map.has(uuid)) {
                       const existingName = map.get(uuid);
                       if (existingName !== name) {
                           console.warn(` UUID partizione duplicato con nomi diversi: ${uuid}`, {
                               existing: existingName,
                               new: name,
                               source: `document ${docIndex}`
                           });
                       }
                   } else {
                       map.set(uuid, name);
                       //console.log(` Partizione da documento mappata: ${uuid} -> ${name}`);
                   }
               });

           } catch (docError) {
               console.error(` Errore processing documento ${docIndex}:`, docError);
               continue;
           }
       }
   }

   // ========================================
   // STRATEGY 4: ENDPOINT PARTITIONS (fallback)
   // ========================================
   if (jsonContent?.endpoints && Array.isArray(jsonContent.endpoints)) {
       console.log(`🔌 Scanning ${jsonContent.endpoints.length} endpoint per partizioni aggiuntive...`);

       let endpointPartitionsFound = 0;

       for (const [endpointIndex, endpoint] of jsonContent.endpoints.entries()) {
           try {
               if (!endpoint || !Array.isArray(endpoint.partitions)) {
                   continue;
               }

               for (const partitionUuid of endpoint.partitions) {
                   if (typeof partitionUuid === 'string' && partitionUuid && !map.has(partitionUuid)) {
                       const fallbackName = `EndpointPartition_${partitionUuid.substring(0, 8)}`;
                       map.set(partitionUuid, fallbackName);
                       //console.log(`🔌 Partizione da endpoint mappata: ${partitionUuid} -> ${fallbackName}`);
                       endpointPartitionsFound++;
                   }
               }
           } catch (endpointError) {
               console.error(` Errore scanning endpoint ${endpointIndex} per partizioni:`, endpointError);
               continue;
           }
       }

       if (endpointPartitionsFound > 0) {
           console.log(` Trovate ${endpointPartitionsFound} partizioni aggiuntive dagli endpoint`);
       }
   }

   // ========================================
   // VALIDAZIONE E REPORT FINALE
   // ========================================
   // console.log("=== REPORT MAPPA GLOBALE PARTIZIONI ===");
   //console.log(` Totale partizioni mappate: ${map.size}`);

   if (map.size === 0) {
       //  console.error(" ERRORE CRITICO: Nessuna partizione trovata nel JSON!");


       /*console.log(" Struttura JSON disponibile:", {
             hasAreas: !!jsonContent?.areas,
             hasPartitions: !!jsonContent?.partitions,
             hasEndpoints: !!jsonContent?.endpoints,
             areasType: typeof jsonContent?.areas,
             partitionsType: typeof jsonContent?.partitions
         });             /
   } else {
   // Mostra le prime 5 partizioni per debug
   const samplePartitions = Array.from(map.entries()).slice(0, 5);
   console.log("🔍 Campione partizioni mappate:", samplePartitions);

   // Statistiche
   const namePatterns = Array.from(map.values()).reduce((patterns: Record<string, number>, name) => {
       if (name.startsWith('Partition_')) patterns.generated = (patterns.generated || 0) + 1;
       else if (name.startsWith('EndpointPartition_')) patterns.endpoint = (patterns.endpoint || 0) + 1;
       else patterns.explicit = (patterns.explicit || 0) + 1;
       return patterns;
   }, {});

   console.log("📈 Tipi di nomi partizioni:", namePatterns);
}

return map;
}*/ 
