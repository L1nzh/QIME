#!/usr/bin/env python3
"""
Simple UMLS Integration for QIME
Direct term-to-definition mapping without complex CUI resolution.
"""

import os
import json
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class SimpleUMLSIntegration:
    def __init__(self, 
                 mrdef_path=None,
                 mrconso_path=None,
                 mrsty_path=None,
                 sab_priority=None,
                 interested_semtypes=None,
                 enable_semtype_filtering=True,
                 preextracted_terms_path=None, qwen_terms_path=None):
        
        # Base directory for relative paths (QIME/framework)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir) # QIME/

        # Default paths relative to project root
        self.mrdef_path = mrdef_path or os.path.join(project_root, "2025AA/META/MRDEF.RRF")
        self.mrconso_path = mrconso_path or os.path.join(project_root, "2025AA/META/MRCONSO.RRF")
        self.mrsty_path = mrsty_path or os.path.join(project_root, "2025AA/META/MRSTY.RRF")
        
        self.sab_priority = sab_priority or ["MSH", "NCI", "CSP", "SNOMEDCT_US"]
        self.enable_semtype_filtering = enable_semtype_filtering
        
        # Comprehensive medical semantic types
        MEDICAL_SEMTYPES = {
            # Diseases and Conditions
            'T047', 'T191', 'T046', 'T048', 'T184',
            # Drugs and Chemicals
            'T121', 'T109', 'T200', 'T195', 'T196', 'T197',
            # Genes and Molecules
            'T028', 'T116', 'T123', 'T087',
            # Physiological and Pathological Processes
            'T040', 'T042', 'T043', 'T044',
            # Anatomy and Body Structures
            'T017', 'T029', 'T023', 'T025',
            # Diagnostic and Therapeutic Procedures
            'T060', 'T061', 'T058', 'T065'
        }
        self.interested_semtypes = interested_semtypes or MEDICAL_SEMTYPES
        
        # Support both parameter names
        self.preextracted_terms_path = preextracted_terms_path or qwen_terms_path
        
        # Load general terms filter list
        self.general_terms = self._load_general_terms()
        
        # Load pre-extracted terms
        self.preextracted_terms_data = self._load_preextracted_terms()
        
        # Build indexes
        if self.enable_semtype_filtering:
            self.cui_to_semtypes = self._build_semantic_type_index()
        else:
            self.cui_to_semtypes = {}
            
        self.term_to_definition = self._build_term_definition_index()
    
    def _load_general_terms(self):
        """Load general terms filter list from JSON file."""
        # Relative path: QIME/data/general_terms.json
        base_dir = os.path.dirname(os.path.abspath(__file__))
        general_terms_path = os.path.join(base_dir, "../data/general_terms.json")
        
        try:
            with open(general_terms_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            terms_set = set()
            for term in data.get('term_list', []):
                terms_set.add(term.lower())
            
            logger.info(f"[INFO] Loaded {len(terms_set)} general terms for filtering")
            return terms_set
            
        except FileNotFoundError:
            logger.warning(f"[WARN] General terms file not found: {general_terms_path}")
            return set()
        except Exception as e:
            logger.warning(f"[WARN] Error loading general terms: {e}")
            return set()
    
    def _load_preextracted_terms(self):
        """Load pre-extracted terms."""
        if not self.preextracted_terms_path or not os.path.exists(self.preextracted_terms_path):
            # If path is relative, try to resolve it from current working dir or project root
            # But usually it's passed as an argument. If it fails, raise error.
            raise FileNotFoundError(f"Pre-extracted terms file not found: {self.preextracted_terms_path}")
        
        logger.info("[INFO] Loading pre-extracted terms...")
        with open(self.preextracted_terms_path, 'r') as f:
            data = json.load(f)
        
        terms_data = data.get('medical_terms', [])
        logger.info(f"[INFO] Loaded terms for {len(terms_data)} documents")
        return terms_data
    
    def _build_semantic_type_index(self):
        """Build CUI to semantic types mapping from MRSTY.RRF."""
        # Cache file in 2025AA directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(base_dir, "../2025AA/cui_semtypes_cache.json")
        
        if os.path.exists(cache_path):
            logger.info("[INFO] Loading cached semantic type index...")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        logger.info("[INFO] Building semantic type index from MRSTY.RRF...")
        cui_to_semtypes = defaultdict(set)
        
        if not os.path.exists(self.mrsty_path):
             logger.warning(f"[WARN] MRSTY file not found: {self.mrsty_path}")
             return {}

        with open(self.mrsty_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 1000000 == 0:
                    logger.info(f"Processed {line_num:,} MRSTY lines...")
                
                fields = line.strip().split('|')
                if len(fields) >= 2:
                    cui = fields[0]
                    semtype = fields[1]
                    cui_to_semtypes[cui].add(semtype)
        
        cui_to_semtypes_serializable = {cui: list(semtypes) for cui, semtypes in cui_to_semtypes.items()}
        
        logger.info(f"[INFO] Built semantic type index for {len(cui_to_semtypes_serializable)} CUIs")
        
        # Cache the results
        try:
            with open(cache_path, 'w') as f:
                json.dump(cui_to_semtypes_serializable, f)
        except Exception as e:
            logger.warning(f"[WARN] Failed to write cache: {e}")
        
        return cui_to_semtypes_serializable
    
    def _build_term_definition_index(self):
        """Build direct term-to-definition mapping from MRCONSO + MRDEF."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.join(base_dir, "../2025AA/simple_term_definition_cache.json")
        
        if os.path.exists(cache_path):
            logger.info("[INFO] Loading term-to-definition index from cache...")
            with open(cache_path, 'r') as f:
                term_to_definition = json.load(f)
            logger.info(f"[INFO] Loaded definitions for {len(term_to_definition)} terms from cache")
            return term_to_definition
        
        logger.info("[INFO] Building term-to-definition index (this may take a few minutes)...")
        
        cui_to_definition = self._build_cui_definition_mapping()
        term_to_cui = self._build_term_cui_mapping()
        
        term_to_definition = {}
        target_sabs = set(self.sab_priority)
        
        for term, cui in term_to_cui.items():
            if cui in cui_to_definition:
                if self.enable_semtype_filtering:
                    cui_semtypes = set(self.cui_to_semtypes.get(cui, []))
                    if not cui_semtypes.intersection(self.interested_semtypes):
                        continue
                
                for sab in self.sab_priority:
                    if sab in cui_to_definition[cui]:
                        definition = cui_to_definition[cui][sab]
                        term_to_definition[term] = {
                            'definition': definition,
                            'source': sab,
                            'cui': cui
                        }
                        break
        
        logger.info(f"[INFO] Built definitions for {len(term_to_definition)} terms")
        
        try:
            logger.info("[INFO] Saving term-to-definition index to cache...")
            with open(cache_path, 'w') as f:
                json.dump(term_to_definition, f)
            logger.info("[INFO] Index cached successfully")
        except Exception as e:
            logger.warning(f"[WARN] Failed to write cache: {e}")
        
        return term_to_definition
    
    def _build_cui_definition_mapping(self):
        """Build CUI-to-definition mapping from MRDEF."""
        cui_to_definition = defaultdict(dict)
        target_sabs = set(self.sab_priority)
        
        if not os.path.exists(self.mrdef_path):
            logger.warning(f"[WARN] MRDEF file not found: {self.mrdef_path}")
            return cui_to_definition
        
        logger.info("[INFO] Building CUI-to-definition mapping from MRDEF...")
        with open(self.mrdef_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, raw in enumerate(f):
                if line_num % 100000 == 0 and line_num > 0:
                    logger.info(f"Processed {line_num} MRDEF lines...")
                
                parts = raw.rstrip("\n").split("|")
                if len(parts) < 6:
                    continue
                
                cui = parts[0]
                sab = parts[4]
                definition = parts[5].strip()
                
                if sab in target_sabs and definition:
                    cui_to_definition[cui][sab] = definition
        
        logger.info(f"[INFO] Built definitions for {len(cui_to_definition)} CUIs")
        return cui_to_definition
    
    def _build_term_cui_mapping(self):
        """Build term-to-CUI mapping from MRCONSO."""
        term_to_cui = {}
        target_sabs = set(self.sab_priority)
        
        if not os.path.exists(self.mrconso_path):
            logger.warning(f"[WARN] MRCONSO file not found: {self.mrconso_path}")
            return term_to_cui
        
        logger.info("[INFO] Building term-to-CUI mapping from MRCONSO...")
        with open(self.mrconso_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, raw in enumerate(f):
                if line_num % 1000000 == 0 and line_num > 0:
                    logger.info(f"Processed {line_num} MRCONSO lines...")
                
                parts = raw.rstrip("\n").split("|")
                if len(parts) < 15:
                    continue
                
                cui = parts[0]
                sab = parts[11]
                term = parts[14].strip().lower()
                
                if sab in target_sabs and term:
                    if term not in term_to_cui:
                        term_to_cui[term] = cui
        
        logger.info(f"[INFO] Built CUI mapping for {len(term_to_cui)} terms")
        return term_to_cui
    
    def extract_cluster_terms_from_indices(self, doc_indices, max_terms=5):
        """Extract top terms with definitions from document indices."""
        all_terms = []
        for doc_idx in doc_indices:
            if doc_idx < len(self.preextracted_terms_data):
                doc_terms = self.preextracted_terms_data[doc_idx]
                if doc_terms:
                    all_terms.extend(doc_terms)
        
        term_counts = Counter(all_terms)
        sorted_terms = term_counts.most_common()
        
        result_terms = []
        seen_definitions = set()
        
        for term, count in sorted_terms:
            if len(result_terms) >= max_terms:
                break
            
            term_lower = term.strip().lower()
            if term_lower in self.general_terms:
                continue
            
            term_info = self.term_to_definition.get(term_lower)
            if not term_info:
                continue
            
            definition = term_info['definition']
            def_lower = definition.lower()
            
            if any(def_lower in seen_def or seen_def in def_lower for seen_def in seen_definitions):
                continue
            
            result_terms.append({
                'term': term,
                'cui': term_info['cui'],
                'definition': definition,
                'source': term_info['source'],
                'frequency': count
            })
            seen_definitions.add(def_lower)
        
        return result_terms
    
    def format_umls_context(self, top_terms):
        """Format UMLS terms for prompt context."""
        if not top_terms:
            return ""
        
        context = "\n\nRelevant medical concepts in positive articles:\n"
        for term_info in top_terms:
            context += f"- {term_info['term']}: {term_info['definition']}\n"
        
        return context
