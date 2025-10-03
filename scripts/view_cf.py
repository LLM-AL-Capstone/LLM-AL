#!/usr/bin/env python3
"""
Simple viewer for generated counterfactual sentences
"""
import argparse
import json
from pathlib import Path

def view_candidates(file_path, limit=10, min_score=0.0):
    """View counterfactual candidates with filtering options"""
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    with open(file_path) as f:
        candidates = json.load(f)
    
    # Filter by score if specified
    if min_score > 0:
        candidates = [c for c in candidates if c.get('score', 0) >= min_score]
    
    print(f"📊 Found {len(candidates)} candidates")
    if min_score > 0:
        print(f"   (filtered for score >= {min_score})")
    
    print("=" * 80)
    
    for i, candidate in enumerate(candidates[:limit]):
        score = candidate.get('score', 0)
        
        print(f"\n🔢 #{i+1} | Score: {score:.3f}")
        print("-" * 40)
        
        print(f"📄 Original ({candidate['original_label']}):")
        print(f"   {candidate['original'][:100]}{'...' if len(candidate['original']) > 100 else ''}")
        
        print(f"✨ Counterfactual ({candidate['counterfactual_label']}):")  
        print(f"   {candidate['counterfactual'][:100]}{'...' if len(candidate['counterfactual']) > 100 else ''}")
        
        # Show filter details if available
        filter_info = candidate.get('filter', candidate.get('judge', {}))
        if filter_info:
            print(f"🔍 Filter Details:")
            for key, value in filter_info.items():
                if key != 'notes':
                    print(f"   {key}: {value}")
        
        print("-" * 40)
    
    if len(candidates) > limit:
        print(f"\n... and {len(candidates) - limit} more candidates")

def main():
    parser = argparse.ArgumentParser(description="View generated counterfactual sentences")
    parser.add_argument("--file", "-f", default="reports/demos/all_candidates_emotions_latest.json", 
                       help="Path to candidates file")
    parser.add_argument("--limit", "-l", type=int, default=10, 
                       help="Number of candidates to show")
    parser.add_argument("--min-score", "-s", type=float, default=0.0,
                       help="Minimum filter score to show")
    parser.add_argument("--list-files", action="store_true",
                       help="List available candidate files")
    
    args = parser.parse_args()
    
    if args.list_files:
        print("📁 Available candidate files:")
        demo_path = Path("reports/demos")
        if demo_path.exists():
            for file in sorted(demo_path.glob("all_candidates_*.json")):
                size = len(json.loads(file.read_text())) if file.exists() else 0
                print(f"   {file.name} ({size} candidates)")
        else:
            print("   No demo files found")
        return
    
    view_candidates(args.file, args.limit, args.min_score)

if __name__ == "__main__":
    main()