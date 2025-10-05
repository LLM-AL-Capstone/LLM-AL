"""
Variation Theory Three-Stage Filtering Pipeline

Implements the complete filtering system from the research paper:
- C1: Heuristic Filter (malformed text detection)
- C2: Neuro-Symbolic Filter (Pattern Keeping Rate - PKR)  
- C3: LLM Discriminator Filter (Label Flip Rate - LFR, Soft Label Flip Rate - SLFR)
"""

import json
import re
from pathlib import Path
from jinja2 import Template
from ..llm.ollama import OllamaClient


class VariationTheoryFilter:
    """Three-stage filtering pipeline matching the research paper"""
    
    def __init__(self, cfg):
        self.client = OllamaClient(cfg["model_ann"], temperature=0.0)
        self.cfg = cfg
        
        # Load prompt templates
        self._load_templates()
    
    def _load_templates(self):
        """Load Jinja2 templates for filtering stages"""
        # Only load combined filter template - individual filters are no longer used
        # Individual filters were: pattern_consistency_filter.txt, label_flip_discriminator.txt, filter.txt
        self.combined_template = Template(Path("prompts/filtering/combined_filter.txt").read_text())
    
    def apply_three_stage_filter(self, original, counterfactual, pattern_info, target_label):
        """Apply C1 + optimized combined C2+C3 filtering pipeline"""
        
        # Stage C1: Regex Heuristic Filter
        c1_result = self._stage_c1_heuristic_filter(counterfactual)
        if not c1_result["pass"]:
            return {
                "pass_all": False,
                "stage_failed": "C1",
                "reason": c1_result["reason"],
                "pkr": 0.0,
                "lfr": 0.0,
                "slfr": 0.0,
                "score": 0.0,
                "details": {"c1": c1_result}
            }
        
        # Combined C2+C3: Use single LLM call for all metrics (OPTIMIZATION)
        combined_result = self._combined_filter(original, counterfactual, pattern_info, target_label)
        
        pkr_score = combined_result["pkr"]
        lfr_score = combined_result["lfr"]
        slfr_score = combined_result["slfr"]
        
        # Pass thresholds from config
        pkr_threshold = self.cfg.get("pkr_threshold", 0.7)
        lfr_threshold = self.cfg.get("lfr_threshold", 0.8)
        slfr_threshold = self.cfg.get("slfr_threshold", 0.8)
        
        # Determine which stage failed (if any)
        stage_failed = "None"
        reason = combined_result.get("reason", "")
        
        if pkr_score < pkr_threshold:
            stage_failed = "C2"
            reason = f"Pattern violation (PKR: {pkr_score:.3f} < {pkr_threshold})"
        elif lfr_score < lfr_threshold or slfr_score < slfr_threshold:
            stage_failed = "C3"
            reason = f"Label flip issues (LFR: {lfr_score:.3f}, SLFR: {slfr_score:.3f})"
        
        pass_all = (pkr_score >= pkr_threshold and 
                   lfr_score >= lfr_threshold and 
                   slfr_score >= slfr_threshold)
        
        # Calculate overall score (weighted combination)
        # PKR: 25%, LFR: 35%, SLFR: 25%, Quality: 15%
        quality_scores = combined_result["quality"]
        quality_avg = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.0
        overall_score = (pkr_score * 0.25 + lfr_score * 0.35 + slfr_score * 0.25 + quality_avg * 0.15)
        
        return {
            "pass_all": pass_all,
            "stage_failed": stage_failed,
            "reason": reason,
            "pkr": pkr_score,
            "lfr": lfr_score, 
            "slfr": slfr_score,
            "score": overall_score,
            "details": {
                "c1": c1_result,
                "combined": combined_result
            }
        }
    
    def _stage_c1_heuristic_filter(self, text):
        """C1: Enhanced heuristic filter for malformed generations"""
        
        # Basic malformation checks
        if len(text.strip()) < 5:
            return {"pass": False, "reason": "Text too short"}
        
        # Check for unmatched quotes
        if text.count('"') % 2 != 0:
            return {"pass": False, "reason": "Unmatched quotes"}
        
        # Check for incomplete/cutoff text
        if text.endswith("...") or text.endswith(".."):
            return {"pass": False, "reason": "Incomplete text (ellipsis)"}
        
        # Check for prompt leakage
        prompt_indicators = [
            "original:", "counterfactual:", "label:", "target:", 
            "instructions:", "example:", "generate", "json:"
        ]
        text_lower = text.lower()
        if any(text_lower.startswith(indicator) for indicator in prompt_indicators):
            return {"pass": False, "reason": "Prompt leakage"}
        
        # Check for repeated patterns (sign of generation issues)
        words = text.split()
        if len(words) > 3:
            # Check for excessive repetition
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            max_repetition = max(word_counts.values()) if word_counts else 0
            if max_repetition > len(words) // 2:
                return {"pass": False, "reason": "Excessive word repetition"}
        
        # Check for malformed JSON remnants
        json_indicators = ["{", "}", "[", "]", "```", "json"]
        if any(indicator in text for indicator in json_indicators):
            return {"pass": False, "reason": "JSON formatting remnants"}
        
        # Check for non-English characters (basic)
        if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text):  # Chinese characters
            return {"pass": False, "reason": "Non-English characters detected"}
        
        return {"pass": True, "reason": "Passed heuristic checks"}
    
    def _stage_c2_pattern_filter(self, text, pattern_info):
        """C2: Neuro-symbolic pattern consistency filter (PKR)"""
        
        if not pattern_info or pattern_info.get("strategy") != "pattern_guided":
            # For general strategy, assume pattern is reasonably preserved
            return {"pkr": 0.8, "reason": "General strategy - no specific pattern"}
        
        pattern_rule = pattern_info.get("pattern_rule", "")
        if not pattern_rule or pattern_rule == "N/A":
            return {"pkr": 0.7, "reason": "No pattern rule available"}
        
        # Render pattern consistency prompt
        prompt = self.pattern_template.render(
            pattern_rule=pattern_rule,
            text=text
        )
        
        try:
            response = self.client.run(prompt, system="You are an English-speaking assistant. Respond only in English.", max_tokens=200, retries=1)
            
            # Parse JSON response
            result = self._parse_json_response(response)
            if result:
                pkr_score = float(result.get("pkr_score", 0.5))
                pkr_score = min(max(pkr_score, 0.0), 1.0)  # Clamp to [0,1]
                reason = result.get("reason", "Pattern consistency evaluated")
                return {"pkr": pkr_score, "reason": reason}
            else:
                # Fallback: extract score from text
                score_match = re.search(r'(\d+\.?\d*)', response)
                if score_match:
                    pkr_score = float(score_match.group(1))
                    if pkr_score > 1.0:  # Handle percentage format
                        pkr_score = pkr_score / 100.0
                    pkr_score = min(max(pkr_score, 0.0), 1.0)
                    return {"pkr": pkr_score, "reason": f"Pattern consistency: {pkr_score:.3f}"}
                else:
                    return {"pkr": 0.5, "reason": "Could not parse PKR score"}
                    
        except Exception as e:
            return {"pkr": 0.5, "reason": f"Error evaluating pattern: {str(e)[:50]}"}
    
    def _stage_c3_discriminator_filter(self, original, counterfactual, target_label):
        """C3: LLM discriminator for label flip verification (LFR & SLFR)"""
        
        # Render discriminator prompt
        prompt = self.discriminator_template.render(
            original=original,
            counterfactual=counterfactual,
            target_label=target_label
        )
        
        try:
            response = self.client.run(prompt, system="You are an English-speaking assistant. Respond only in English.", max_tokens=200, retries=1)
            
            # Parse JSON response
            result = self._parse_json_response(response)
            if result:
                lfr_score = float(result.get("lfr_score", 0.0))
                slfr_score = float(result.get("slfr_score", 0.0))
                reason = result.get("reason", "Label flip evaluated")
                
                # Clamp scores to [0,1]
                lfr_score = min(max(lfr_score, 0.0), 1.0)
                slfr_score = min(max(slfr_score, 0.0), 1.0)
                
                return {
                    "lfr": lfr_score,
                    "slfr": slfr_score,
                    "reason": reason
                }
            else:
                # Fallback: parse from text
                lfr_match = re.search(r'LFR[:\s]*(\d+\.?\d*)', response, re.IGNORECASE)
                slfr_match = re.search(r'SLFR[:\s]*(\d+\.?\d*)', response, re.IGNORECASE)
                
                lfr_score = float(lfr_match.group(1)) if lfr_match else 0.0
                slfr_score = float(slfr_match.group(1)) if slfr_match else 0.0
                
                # Handle percentage format
                if lfr_score > 1.0:
                    lfr_score = lfr_score / 100.0
                if slfr_score > 1.0:
                    slfr_score = slfr_score / 100.0
                
                # Clamp scores to [0,1]
                lfr_score = min(max(lfr_score, 0.0), 1.0)
                slfr_score = min(max(slfr_score, 0.0), 1.0)
                
                return {
                    "lfr": lfr_score,
                    "slfr": slfr_score,
                    "reason": "Label flip parsed from text"
                }
                
        except Exception as e:
            return {
                "lfr": 0.0,
                "slfr": 0.0,
                "reason": f"Error in discriminator evaluation: {str(e)[:50]}"
            }
    
    def _stage_c3_quality_filter(self, original, counterfactual, target_label):
        """C3: Additional quality assessment (minimality, fluency, etc.)"""
        
        # Render quality assessment prompt
        prompt = self.quality_template.render(
            orig=original,
            cf=counterfactual,
            target=target_label
        )
        
        try:
            response = self.client.run(prompt, system="You are an English-speaking assistant. Respond only in English.", max_tokens=200, retries=1)
            
            # Parse JSON response
            result = self._parse_json_response(response)
            if result:
                # Extract quality scores
                quality_scores = {
                    "minimality": float(result.get("minimality", 0.0)),
                    "fluency": float(result.get("fluency", 0.0)),
                    "label_determinism": float(result.get("label_determinism", 0.0)),
                    "faithfulness": float(result.get("faithfulness", 0.0))
                }
                
                # Clamp all scores to [0,1]
                for key in quality_scores:
                    quality_scores[key] = min(max(quality_scores[key], 0.0), 1.0)
                
                return quality_scores
            else:
                # Return default scores if parsing fails
                return {
                    "minimality": 0.5,
                    "fluency": 0.5,
                    "label_determinism": 0.5,
                    "faithfulness": 0.5
                }
                
        except Exception as e:
            return {
                "minimality": 0.5,
                "fluency": 0.5,
                "label_determinism": 0.5,
                "faithfulness": 0.5
            }
    
    def _combined_filter(self, original, counterfactual, pattern_info, target_label):
        """Combined C2+C3 filter - evaluates all metrics in single LLM call (OPTIMIZATION)"""
        
        pattern_rule = ""
        if pattern_info and pattern_info.get("strategy") == "pattern_guided":
            pattern_rule = pattern_info.get("pattern_rule", "")
        
        # Render combined filter prompt
        prompt = self.combined_template.render(
            original=original,
            counterfactual=counterfactual,
            pattern_rule=pattern_rule,
            target_label=target_label
        )
        
        try:
            response = self.client.run(prompt, system="You are an English-speaking assistant. Respond only in English.", max_tokens=400, retries=1)
            
            # Parse JSON response
            result = self._parse_json_response(response)
            if result:
                # Extract all scores
                pkr_score = float(result.get("pkr_score", 0.5))
                lfr_score = float(result.get("lfr_score", 0.0))
                slfr_score = float(result.get("slfr_score", 0.0))
                minimality = float(result.get("minimality", 0.5))
                fluency = float(result.get("fluency", 0.5))
                label_determinism = float(result.get("label_determinism", 0.5))
                faithfulness = float(result.get("faithfulness", 0.5))
                reason = result.get("overall_reason", "Combined evaluation completed")
                
                # Clamp all scores to [0,1]
                pkr_score = min(max(pkr_score, 0.0), 1.0)
                lfr_score = min(max(lfr_score, 0.0), 1.0)
                slfr_score = min(max(slfr_score, 0.0), 1.0)
                minimality = min(max(minimality, 0.0), 1.0)
                fluency = min(max(fluency, 0.0), 1.0)
                label_determinism = min(max(label_determinism, 0.0), 1.0)
                faithfulness = min(max(faithfulness, 0.0), 1.0)
                
                return {
                    "pkr": pkr_score,
                    "lfr": lfr_score,
                    "slfr": slfr_score,
                    "quality": {
                        "minimality": minimality,
                        "fluency": fluency,
                        "label_determinism": label_determinism,
                        "faithfulness": faithfulness
                    },
                    "reason": reason
                }
            else:
                # Fallback: parse individual scores from text
                pkr_match = re.search(r'pkr[_\s]*score[:\s]*(\d+\.?\d*)', response, re.IGNORECASE)
                lfr_match = re.search(r'lfr[_\s]*score[:\s]*(\d+\.?\d*)', response, re.IGNORECASE)
                slfr_match = re.search(r'slfr[_\s]*score[:\s]*(\d+\.?\d*)', response, re.IGNORECASE)
                
                pkr_score = float(pkr_match.group(1)) if pkr_match else 0.5
                lfr_score = float(lfr_match.group(1)) if lfr_match else 0.0
                slfr_score = float(slfr_match.group(1)) if slfr_match else 0.0
                
                # Handle percentage format
                if pkr_score > 1.0:
                    pkr_score = pkr_score / 100.0
                if lfr_score > 1.0:
                    lfr_score = lfr_score / 100.0
                if slfr_score > 1.0:
                    slfr_score = slfr_score / 100.0
                
                # Clamp scores to [0,1]
                pkr_score = min(max(pkr_score, 0.0), 1.0)
                lfr_score = min(max(lfr_score, 0.0), 1.0)
                slfr_score = min(max(slfr_score, 0.0), 1.0)
                
                return {
                    "pkr": pkr_score,
                    "lfr": lfr_score,
                    "slfr": slfr_score,
                    "quality": {
                        "minimality": 0.5,
                        "fluency": 0.5,
                        "label_determinism": 0.5,
                        "faithfulness": 0.5
                    },
                    "reason": "Combined evaluation parsed from text"
                }
                
        except Exception as e:
            return {
                "pkr": 0.5,
                "lfr": 0.0,
                "slfr": 0.0,
                "quality": {
                    "minimality": 0.5,
                    "fluency": 0.5,
                    "label_determinism": 0.5,
                    "faithfulness": 0.5
                },
                "reason": f"Error in combined evaluation: {str(e)[:50]}"
            }

    def _parse_json_response(self, response):
        """Parse JSON from LLM response with fallback handling"""
        try:
            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()
            
            # Find JSON object
            i = response.find("{")
            j = response.rfind("}")
            
            if i != -1 and j != -1 and j > i:
                json_str = response[i:j+1]
                return json.loads(json_str)
            
        except json.JSONDecodeError:
            pass
        
        return None