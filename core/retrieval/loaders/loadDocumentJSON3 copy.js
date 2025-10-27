"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.loadDocumentsJSON = loadDocumentsJSON;
exports.buildAreaPartitionMaps = buildAreaPartitionMaps;
var RAG_js_1 = require("../../../config/RAG.js");
var fs_1 = require("fs");
var document_1 = require("langchain/document"); //Document non è un array e non ha un metodo push
var buildGlobalPartitionsMap_js_1 = require("./buildGlobalPartitionsMap.js");
function loadDocumentsJSON() {
    return __awaiter(this, void 0, void 0, function () {
        var processedUUIDs, rawContent, error_1, jsonContent, hasValidEndpoints, hasValidAreas, globalPartitionMap, areaPartitionMaps, endpointAreaRelations, Documents, installationContent, mainDocument;
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m;
        return __generator(this, function (_o) {
            switch (_o.label) {
                case 0:
                    processedUUIDs = new Set();
                    _o.label = 1;
                case 1:
                    _o.trys.push([1, 3, , 4]);
                    return [4 /*yield*/, fs_1.promises.readFile(RAG_js_1.filePath, 'utf-8')];
                case 2:
                    rawContent = _o.sent();
                    return [3 /*break*/, 4];
                case 3:
                    error_1 = _o.sent();
                    console.error("ERROR: Document not found or not readable!");
                    throw new Error("Execution blocked: file doesn't exist");
                case 4:
                    if (!rawContent || rawContent.trim().length === 0) {
                        console.error("ERROR: Empty document!");
                        throw new Error("Execution blocked: file contents empty.");
                    }
                    try {
                        jsonContent = JSON.parse(rawContent);
                    }
                    catch (parseError) {
                        console.error("ERROR: JSON parsing failed", parseError);
                        throw new Error("Invalid JSON format in configuration");
                    }
                    if (!jsonContent || typeof jsonContent !== 'object') {
                        throw new Error("Invalid JSON structure: root must be an object");
                    }
                    hasValidEndpoints = Array.isArray(jsonContent.endpoints) && jsonContent.endpoints.length > 0;
                    hasValidAreas = Array.isArray(jsonContent.areas) && jsonContent.areas.length > 0;
                    if (!hasValidEndpoints) {
                        console.warn("No valid endpoint found in JSON content");
                        return [2 /*return*/, getFallbackDocument(new Error("No valid endpoints in configuration"))];
                    }
                    console.log("Data structure: ".concat(jsonContent.endpoints.length, " endpoints, ").concat(((_a = jsonContent.areas) === null || _a === void 0 ? void 0 : _a.length) || 0, " area"));
                    globalPartitionMap = (0, buildGlobalPartitionsMap_js_1.buildGlobalPartitionMap)(jsonContent);
                    areaPartitionMaps = hasValidAreas ? buildAreaPartitionMaps(jsonContent) : [];
                    endpointAreaRelations = hasValidAreas ? buildEndpointAreaRelations(jsonContent, areaPartitionMaps) : new Map();
                    Documents = [];
                    installationContent = {
                        type: "installation-config",
                        metadata: jsonContent.metadata || {},
                        statistics: {
                            totalEndpoints: ((_b = jsonContent.endpoints) === null || _b === void 0 ? void 0 : _b.length) || 0,
                            totalAreas: ((_c = jsonContent.areas) === null || _c === void 0 ? void 0 : _c.length) || 0,
                            totalPartitions: globalPartitionMap.size,
                            sensorCount: jsonContent.endpoints.filter(function (ep) { return ep.category === 18; }).length || 0,
                            actuatorCount: ((_d = jsonContent.endpoints) === null || _d === void 0 ? void 0 : _d.filter(function (ep) { return [11, 12, 15].includes(ep.category); }).length) || 0,
                            controllerCount: ((_e = jsonContent.endpoints) === null || _e === void 0 ? void 0 : _e.filter(function (ep) { return [0, 1, 2].includes(ep.category); }).length) || 0
                        },
                        endpoints: jsonContent.endpoints,
                        areas: jsonContent.areas,
                        globalPartitionMap: Object.fromEntries(globalPartitionMap)
                    };
                    mainDocument = new document_1.Document({
                        pageContent: JSON.stringify(installationContent),
                        metadata: {
                            source: RAG_js_1.targetFile,
                            loc: RAG_js_1.filePath,
                            type: 'intallation-config',
                            isValid: true,
                            timestamp: new Date().toISOString(),
                            name: (_f = jsonContent.metadata) === null || _f === void 0 ? void 0 : _f.name,
                            chunkType: 'summary',
                            installationName: ((_g = jsonContent.metadata) === null || _g === void 0 ? void 0 : _g.name) || 'installation-config',
                            revision: (_h = jsonContent.metadata) === null || _h === void 0 ? void 0 : _h.revision,
                            deviceType: 'installation',
                            totalEndpoints: ((_j = jsonContent.endpoints) === null || _j === void 0 ? void 0 : _j.length) || 0,
                            totalAreas: ((_k = jsonContent.areas) === null || _k === void 0 ? void 0 : _k.length) || 0,
                            hasPartitions: globalPartitionMap.size > 0,
                            hasAreaInfo: hasValidAreas,
                            major: (_l = jsonContent.metadata) === null || _l === void 0 ? void 0 : _l.major,
                            minor: (_m = jsonContent.metadata) === null || _m === void 0 ? void 0 : _m.minor,
                        }
                    });
                    Documents.push(mainDocument);
                    console.log("Single raw document created for two-stage chunking");
                    return [2 /*return*/, {
                            Documents: Documents,
                            partitionMap: globalPartitionMap
                        }];
            }
        });
    });
}
/**Fundamental function for:
Creates the mapping between areas and partitions
Manages both objects and UUID strings for partitions
Has robust validation
Required to resolve partition names */
function buildAreaPartitionMaps(jsonContent) {
    var maps = [];
    if (!jsonContent.areas || !Array.isArray(jsonContent.areas)) {
        console.warn("No areas found in JSON file");
        return maps;
    }
    for (var _i = 0, _a = jsonContent.areas.entries(); _i < _a.length; _i++) {
        var _b = _a[_i], index = _b[0], area = _b[1];
        try {
            // RIGOROUS VALIDATION AREAendpointDoc
            if (!area || typeof area !== 'object') {
                console.warn("Areas ".concat(index, " invalid:"), area);
                continue;
            }
            if (!area.uuid || !area.name) {
                console.warn(" Areas ".concat(index, " with UUID or name missing:"), {
                    uuid: area.uuid,
                    name: area.name
                });
                continue;
            }
            var areaMap = {
                areaUuid: area.uuid,
                areaName: area.name,
                partitions: []
            };
            // SECURE PARTITION PROCESSING
            if (Array.isArray(area.partitions)) {
                for (var _c = 0, _d = area.partitions.entries(); _c < _d.length; _c++) {
                    var _e = _d[_c], partIndex = _e[0], partition = _e[1];
                    if (!partition) {
                        console.warn(" Partizione ".concat(partIndex, " in area ").concat(area.name, " \u00E8 null/undefined"));
                        continue;
                    }
                    // I Handle both objects and UUID strings
                    var partitionUuid = typeof partition === 'string' ? partition : partition.uuid;
                    var partitionName = typeof partition === 'string' ?
                        "Partition_".concat(partition.substring(0, 8)) : partition.name;
                    if (partitionUuid && partitionName) {
                        areaMap.partitions.push({
                            uuid: partitionUuid,
                            name: partitionName
                        });
                    }
                    else {
                        console.warn(" Partition ".concat(partIndex, " in area ").concat(area.name, " with incomplete data"));
                    }
                }
            }
            maps.push(areaMap);
            console.log(" Mapped Area: ".concat(area.name, " (").concat(areaMap.partitions.length, " partitions)"));
        }
        catch (error) {
            console.error(" Error processing area ".concat(index, ":"), error);
            continue;
        }
    }
    console.log("Area maps created: ".concat(maps.length));
    return maps;
}
//Itera sugli endpoints e trova le aree attraverso partizioni condivise
function buildEndpointAreaRelations(jsonContent, areaPartitionMaps) {
    var relations = new Map();
    console.log("  Costruzione relazioni endpoint-area...");
    // Validazione input
    if (!(jsonContent === null || jsonContent === void 0 ? void 0 : jsonContent.areas) || !Array.isArray(jsonContent.areas)) {
        console.warn("Nessuna area nel JSON per costruire relazioni");
        return relations;
    }
    if (!Array.isArray(areaPartitionMaps) || areaPartitionMaps.length === 0) {
        console.warn(" Nessuna mappa partizioni per costruire relazioni");
        return relations;
    }
    console.log("Aree da processare: ".concat(jsonContent.areas.length, ", Mappe partizioni: ").concat(areaPartitionMaps.length));
    console.log("Aree processate:", JSON.stringify(jsonContent.areas), "Nome partizoni", JSON.stringify(areaPartitionMaps));
    var totalEndpointsProcessed = 0;
    var totalRelationsCreated = 0;
    var _loop_1 = function (endpointIndex, endpoint) {
        try {
            totalEndpointsProcessed++;
            // Validazione endpoint
            if (!endpoint || !endpoint.uuid) {
                console.warn("Endpoint ".concat(endpointIndex, " non valido o senza UUID"));
                return "continue";
            }
            // Se l'endpoint non ha partizioni, salta
            if (!Array.isArray(endpoint.partitions) || endpoint.partitions.length === 0) {
                return "continue";
            }
            var endpointName = endpoint.name || "Device_".concat(endpoint.uuid.substring(0, 8));
            // Per ogni area, controlla se condivide partizioni con questo endpoint
            for (var _f = 0, areaPartitionMaps_1 = areaPartitionMaps; _f < areaPartitionMaps_1.length; _f++) {
                var areaMap = areaPartitionMaps_1[_f];
                // Trova partizioni in comune tra endpoint e area
                var sharedPartitions = areaMap.partitions.filter(function (areaPartition) {
                    return endpoint.partitions.includes(areaPartition.uuid);
                });
                // Se ci sono partizioni condivise, crea la relazione
                if (sharedPartitions.length > 0) {
                    var relation = {
                        endpointUuid: endpoint.uuid,
                        endpointName: endpointName,
                        areaUuid: areaMap.areaUuid,
                        areaName: areaMap.areaName,
                        partitionUuids: sharedPartitions.map(function (p) { return p.uuid; }),
                        location: sharedPartitions.map(function (p) { return p.name; })
                    };
                    // Verifica duplicati (un endpoint può essere in più aree)
                    if (relations.has(endpoint.uuid)) {
                        var existing = relations.get(endpoint.uuid);
                        console.log("Endpoint ".concat(endpoint.uuid, " gi\u00E0 mappato a ").concat(existing === null || existing === void 0 ? void 0 : existing.areaName, ", aggiungendo anche ").concat(areaMap.areaName));
                        // In questo caso, potresti voler gestire relazioni multiple
                        // Per ora, manteniamo solo la prima relazione trovata
                    }
                    else {
                        relations.set(endpoint.uuid, relation);
                        totalRelationsCreated++;
                        console.log("Relazione creata: ".concat(endpointName, " -> ").concat(areaMap.areaName, " (").concat(sharedPartitions.length, " partizioni condivise)"));
                    }
                }
            }
        }
        catch (endpointError) {
            console.error("Errore processing endpoint ".concat(endpointIndex, ":"), endpointError);
            return "continue";
        }
    };
    // for (const [areaIndex, area] of jsonContent.areas.entries()) {
    for (var _i = 0, _a = jsonContent.endpoints.entries(); _i < _a.length; _i++) {
        var _b = _a[_i], endpointIndex = _b[0], endpoint = _b[1];
        _loop_1(endpointIndex, endpoint);
    }
    // REPORT FINALE
    console.log("\nREPORT RELAZIONI ENDPOINT-AREA:");
    console.log("Endpoints processati: ".concat(totalEndpointsProcessed));
    console.log("Relazioni create: ".concat(totalRelationsCreated));
    console.log("Relazioni uniche nella mappa: ".concat(relations.size));
    // DEBUG DETTAGLIATO
    if (relations.size > 0) {
        console.log("\nPrime 3 relazioni create:");
        var count = 0;
        for (var _c = 0, _d = relations.entries(); _c < _d.length; _c++) {
            var _e = _d[_c], uuid = _e[0], relation = _e[1];
            if (count >= 3)
                break;
            console.log("   ".concat(count + 1, ". ").concat(relation.endpointName, " -> ").concat(relation.areaName));
            console.log("      UUID: ".concat(uuid));
            console.log("      Partizioni condivise: ".concat(relation.location.join(', ')));
            count++;
        }
    }
    else {
        console.warn("\nATTENZIONE: Nessuna relazione creata!");
        console.log("Debug: verifica che aree e endpoints condividano partizioni");
        // Debug delle partizioni per area
        console.log("Partizioni per area:");
        areaPartitionMaps.forEach(function (areas) {
            console.log("   ".concat(areas.areaName, ": [").concat(areas.partitions.map(function (p) { return p.name; }).join(', '), "]"));
        });
        // Debug delle partizioni per endpoint (primi 5)
        console.log("Partizioni per endpoint (primi 5):");
        jsonContent.endpoints.slice(0, 5).forEach(function (ep) {
            var _a;
            if (((_a = ep.partitions) === null || _a === void 0 ? void 0 : _a.length) > 0) {
                console.log("   ".concat(ep.name, ": [").concat(ep.partitions.join(', '), "]"));
            }
        });
    }
    return relations;
}
function getFallbackDocument(error) {
    var fallbackUUID = 'fallback-' + Math.random().toString(36).substring(2, 9);
    var fallBack = {
        pageContent: JSON.stringify({
            error: "Failed to load document",
            message: error instanceof Error ? error.message : String(error),
            fallbackType: "empty_system"
        }),
        metadata: {
            source: 'fallback',
            loc: 'internal',
            type: 'installation-config', //fallback
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
            chunkStrategy: 'standard',
            chunkType: 'fallback',
            hasAreaInfo: true
        },
        readableText: "System temporarily unavailable. Please check the configuration and try again."
    };
    //"Document loading failed. Please check the configuration file."
    return {
        Documents: [fallBack],
        partitionMap: new Map()
    };
}
