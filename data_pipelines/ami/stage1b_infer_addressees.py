#!/usr/bin/env python3
"""
Stage 1b: Infer Addressees with Gemini (Using API Key)

SIMPLIFIED VERSION - Uses Google AI Studio API Key (no GCP project needed)

Get your API key:
1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key and paste it below in the API_KEY variable (line 41)

Goal:
    Expand the dataset by inferring addressees for dialogue acts that don't have
    explicit addressee annotations. Uses Gemini API directly with API key.

Features:
    - LLM-based addressee inference with structured reasoning
    - Confidence scores (0-10) for quality filtering
    - Explainability: each inference includes reasoning statement
    - Command-line flags for testing (--meeting, --max-turns)

Input:
    - AMI Corpus dialogue acts (without explicit addressee)
    - stage1_sequences.json (for comparison/statistics)

Output:
    - stage1b_inferred_sequences.json - sequences with inferred addressees
      Each includes: addressees, confidence (0-10), and reasoning
    - stage1_combined_sequences.json - explicit + inferred merged
"""

import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

from config import GEMINI_API_KEY, AMI_CORPUS_DIR, JSON_DUMPS_DIR

try:
    import google.generativeai as genai
except ImportError:
    print("ERROR: Google AI SDK not installed!")
    print("Install with: pip install google-generativeai")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = GEMINI_API_KEY
AMI_CORPUS_DIR = str(AMI_CORPUS_DIR)
JSON_DUMPS_DIR.mkdir(exist_ok=True)

EXPLICIT_SEQUENCES_FILE = JSON_DUMPS_DIR / 'stage1_sequences.json'
OUTPUT_FILE = JSON_DUMPS_DIR / 'stage1b_inferred_sequences.json'
COMBINED_OUTPUT = JSON_DUMPS_DIR / 'stage1_combined_sequences.json'

# Model configuration
# Common working names: 'gemini-pro', 'gemini-1.5-flash', 'gemini-1.5-pro'
# Run test_available_models.py to see which models work with your API key
MODEL_NAME = 'gemini-flash-latest'  # Most widely available and stable

# Inference parameters
CONFIDENCE_THRESHOLD = 6  # Only use predictions with confidence >= 6/10
                          # Lower = more data but lower quality
                          # Higher = less data but higher quality
                          # Recommended: 5-7 for balanced results
MAX_CONTEXT_TURNS = 10  # Number of previous turns to include in prompt
BATCH_SIZE = 10  # Process 10 turns before API call delay
API_DELAY = 1.0  # Seconds to wait between batches (rate limiting)

# ============================================================================
# GEMINI SETUP (API KEY METHOD)
# ============================================================================

def init_gemini(api_key: str):
    """
    Initialize Gemini with API key
    
    Args:
        api_key: Google AI Studio API key
        
    Returns:
        GenerativeModel instance
    """
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        print("\n" + "="*70)
        print("ERROR: API KEY NOT SET!")
        print("="*70)
        print("\nPlease follow these steps:")
        print("1. Go to: https://aistudio.google.com/app/apikey")
        print("2. Click 'Create API Key' (or 'Get API Key')")
        print("3. Copy the key (starts with 'AIza...')")
        print("4. Open this file and paste it on line 39:")
        print("   API_KEY = 'AIza...'")
        print("\nOr set environment variable:")
        print("   export GEMINI_API_KEY='AIza...'")
        print("="*70 + "\n")
        exit(1)
    
    print(f"Initializing Gemini API...")
    print(f"  Model: {MODEL_NAME}")
    
    genai.configure(api_key=api_key)
    
    # Configure model for structured output
    generation_config = {
        'temperature': 0.0,  # Low temperature for consistent output
        'top_p': 0.8,
        'top_k': 20,
        'max_output_tokens': 500,  # Enough for addressees + confidence + reasoning
    }
    
    model = genai.GenerativeModel(MODEL_NAME)
    print("✓ Gemini model initialized")
    
    return model, generation_config


# ============================================================================
# PROMPT ENGINEERING
# ============================================================================

def format_context_for_prompt(context_turns: List[Dict]) -> str:
    """Format context turns for Gemini prompt"""
    if not context_turns:
        return "(No previous context)"
    
    lines = []
    for turn in context_turns:
        lines.append(f"Speaker {turn['speaker']}: {turn['text']}")
    
    return "\n".join(lines)


def create_addressee_inference_prompt(context_turns: List[Dict], 
                                      current_turn: Dict, 
                                      all_speakers: List[str]) -> str:
    """
    Create prompt for Gemini to infer who is being addressed
    
    Returns:
        Prompt string for Gemini
    """
    context_str = format_context_for_prompt(context_turns)
    speakers_str = ', '.join(sorted(all_speakers))
    
    prompt = f"""You are analyzing a multi-party meeting conversation to determine who is being addressed.

SPEAKERS IN MEETING: {speakers_str}

PREVIOUS CONVERSATION:
{context_str}

CURRENT UTTERANCE:
Speaker {current_turn['speaker']}: "{current_turn['text']}"

TASK: Determine who Speaker {current_turn['speaker']} is addressing in the current utterance.

RULES:
- Reply "ALL" if addressing everyone in the meeting
- Reply "NONE" if unclear, rhetorical, or talking to self/thinking aloud
- Otherwise list specific speaker letter(s) separated by commas (e.g., "A" or "A,B,C")
- Consider: name mentions, questions, previous conversation flow, context

OUTPUT FORMAT (must follow exactly):
ADDRESSEES: [speaker letters or ALL or NONE]
CONFIDENCE: [0-10, where 0=completely unsure, 10=absolutely certain]
REASON: [one sentence explaining why you chose these addressees]

EXAMPLES:

Example 1:
ADDRESSEES: A
CONFIDENCE: 9
REASON: Direct name mention and direct question to specific person.

Example 2:
ADDRESSEES: B,C
CONFIDENCE: 7
REASON: Uses "you both" referring to the two people who just spoke.

Example 3:
ADDRESSEES: ALL
CONFIDENCE: 8
REASON: Uses "everyone" and asks a general group question.

Example 4:
ADDRESSEES: NONE
CONFIDENCE: 6
REASON: Appears to be thinking aloud, no clear addressee indicated.

YOUR ANSWER:"""
    
    return prompt


def parse_gemini_response(response_text: str, 
                          all_speakers: List[str]) -> Tuple[List[str], int, str]:
    """
    Parse Gemini's response to extract addressees, confidence, and reasoning
    
    Args:
        response_text: Raw text from Gemini in format:
            ADDRESSEES: A,B
            CONFIDENCE: 8
            REASON: ...
        all_speakers: Valid speaker IDs for validation
        
    Returns:
        (addressees, confidence_0_10, reason) tuple
    """
    response_text = response_text.strip()
    
    # Initialize defaults
    addressees = []
    confidence = 0
    reason = "No explanation provided"
    
    # Extract ADDRESSEES line
    addressees_match = re.search(r'ADDRESSEES:\s*(.+)', response_text, re.IGNORECASE)
    if addressees_match:
        addressees_str = addressees_match.group(1).strip().upper()
        
        if addressees_str == "NONE" or not addressees_str:
            addressees = []
        elif addressees_str == "ALL":
            addressees = all_speakers
        else:
            # Parse comma-separated speaker letters
            addressees_str = re.sub(r'[^A-Z,]', '', addressees_str)
            addressees = [s.strip() for s in addressees_str.split(',') if s.strip()]
            # Validate speakers exist in meeting
            addressees = [s for s in addressees if s in all_speakers]
    
    # Extract CONFIDENCE line (0-10)
    confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', response_text, re.IGNORECASE)
    if confidence_match:
        confidence = int(confidence_match.group(1))
        # Clamp to 0-10 range
        confidence = max(0, min(10, confidence))
    
    # Extract REASON line
    reason_match = re.search(r'REASON:\s*(.+)', response_text, re.IGNORECASE)
    if reason_match:
        reason = reason_match.group(1).strip()
    
    return addressees, confidence, reason


# ============================================================================
# EXTRACTION WITH GEMINI INFERENCE
# ============================================================================

def is_high_quality_dact_text(text: str) -> bool:
    """
    Lightweight quality filter for dialogue acts BEFORE Gemini calls.
    Mirrors Stage 4.5 logic in spirit:
      - drop fillers (mm-hmm, uh, um, etc.)
      - drop very short / mostly punctuation utterances
    """
    import re

    if not text or not text.strip():
        return False

    t = text.strip().lower()

    # Simple filler patterns (subset of Stage 4.5 FILLER_PATTERNS)
    filler_patterns = [
        r'^\s*mm+\s*[-.]?\s*hm+\s*\.?\s*$',  # Mm-hmm, Mmhmm, etc.
        r'^\s*uh+\s*[-.]?\s*huh+\s*\.?\s*$',  # Uh-huh
        r'^\s*um+\s*\.?\s*$',                # Um
        r'^\s*uh+\s*\.?\s*$',                # Uh
        r'^\s*er+\s*\.?\s*$',                # Er
        r'^\s*ah+\s*\.?\s*$',                # Ah
        r'^\s*oh+\s*\.?\s*$',                # Oh
        r'^\s*hm+\s*\.?\s*$',                # Hmm
        r'^\s*\.\s*$',                       # Just punctuation
    ]

    for pattern in filler_patterns:
        if re.match(pattern, t, re.IGNORECASE):
            return False

    # Too short after removing punctuation
    text_alphanum = re.sub(r'[^\w\s]', '', text)
    if len(text_alphanum.strip()) < 3:
        return False

    # Mostly punctuation (>50% non-alphanumeric)
    if len(text_alphanum) < len(text) * 0.5:
        return False

    return True


def load_all_dialogue_acts_without_addressee(corpus_dir: str,
                                             meeting_ids: List[str],
                                             stride: int = 1,
                                             max_samples_per_meeting: Optional[int] = None
                                             ) -> Dict[str, List[Dict]]:
    """
    Load all dialogue acts from AMI corpus that DON'T have addressee annotations
    
    Returns:
        Dict mapping meeting_id -> list of dialogue acts
    """
    import xml.etree.ElementTree as ET
    from stage1_extract_dialogues import extract_dialogue_act_text_and_time
    
    corpus_path = Path(corpus_dir)
    dialogue_acts_dir = corpus_path / 'dialogueActs'
    words_dir = corpus_path / 'words'
    
    meeting_dacts: Dict[str, List[Dict]] = {}
    
    for meeting_id in meeting_ids:
        print(f"Loading dialogue acts from {meeting_id}...")
        
        all_dacts: List[Dict] = []
        
        for da_file in dialogue_acts_dir.glob(f'{meeting_id}.*.dialog-act.xml'):
            speaker = da_file.stem.split('.')[1]
            
            tree = ET.parse(da_file)
            ns = {'nite': 'http://nite.sourceforge.net/'}
            
            for dact in tree.findall('.//dact', ns):
                addressee = dact.get('addressee', '').strip()
                
                # Only process dialogue acts WITHOUT addressee annotation
                if addressee:
                    continue
                
                dact_data = extract_dialogue_act_text_and_time(dact, words_dir, meeting_id, speaker)

                if dact_data and dact_data['text'] and is_high_quality_dact_text(dact_data['text']):
                    all_dacts.append({
                        'meeting_id': meeting_id,
                        'speaker': speaker,
                        'text': dact_data['text'],
                        'starttime': dact_data['starttime'],
                        'endtime': dact_data['endtime'],
                        'dact_id': dact.get('{http://nite.sourceforge.net/}id', ''),
                    })
        
        # Sort by time
        all_dacts.sort(key=lambda x: (x['starttime'], x['speaker']))

        # Apply stride sampling to get disjoint, spaced-out high-quality acts
        if stride > 1:
            all_dacts = all_dacts[::stride]

        # Optionally cap number of samples per meeting
        if max_samples_per_meeting is not None and len(all_dacts) > max_samples_per_meeting:
            all_dacts = all_dacts[:max_samples_per_meeting]

        meeting_dacts[meeting_id] = all_dacts

        print(f"  Loaded {len(all_dacts)} high-quality dialogue acts without addressee")
    
    return meeting_dacts


def get_meeting_speakers(corpus_dir: str, meeting_id: str) -> List[str]:
    """Get all speakers in a meeting"""
    from stage1_extract_dialogues import load_meeting_speakers
    
    meeting_speakers = load_meeting_speakers(corpus_dir)
    
    if meeting_id in meeting_speakers:
        return sorted(list(meeting_speakers[meeting_id].keys()))
    
    return ['A', 'B', 'C', 'D']


def infer_sequences_with_gemini(corpus_dir: str,
                                meeting_ids: List[str],
                                model,
                                generation_config,
                                max_turns_per_meeting: Optional[int] = None,
                                stride: int = 1,
                                max_samples_per_meeting: Optional[int] = None) -> List[Dict]:
    """
    Extract sequences by inferring addressees with Gemini
    """
    meeting_dacts = load_all_dialogue_acts_without_addressee(
        corpus_dir,
        meeting_ids,
        stride=stride,
        max_samples_per_meeting=max_samples_per_meeting,
    )
    sequences = []
    total_api_calls = 0
    
    for meeting_id, all_dacts in meeting_dacts.items():
        print(f"\nProcessing {meeting_id} with Gemini inference...")
        meeting_speakers = get_meeting_speakers(corpus_dir, meeting_id)
        print(f"  Speakers: {', '.join(meeting_speakers)}")
        
        # Limit turns if specified (for testing)
        if max_turns_per_meeting:
            all_dacts = all_dacts[:max_turns_per_meeting]
            print(f"  Limited to first {len(all_dacts)} dialogue acts (test mode)")
        else:
            print(f"  Un-annotated dialogue acts: {len(all_dacts)}")
        
        batch_count = 0
        
        for i, dact in enumerate(all_dacts):
            # Get context
            context_start = max(0, i - MAX_CONTEXT_TURNS)
            context = all_dacts[context_start:i]
            
            # Build current turn
            current_turn = {
                'speaker': dact['speaker'],
                'text': dact['text']
            }
            
            # Create prompt
            prompt = create_addressee_inference_prompt(
                context, current_turn, meeting_speakers
            )
            
            try:
                # Call Gemini API
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                total_api_calls += 1
                
                # Check if response has valid content
                if not response.candidates:
                    # No candidates returned (likely blocked or empty)
                    continue
                
                candidate = response.candidates[0]
                
                # Check finish reason
                # 1=STOP (normal), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
                if candidate.finish_reason not in [1, 'STOP']:
                    # Response was blocked or incomplete
                    # This is normal - just skip this turn
                    continue
                
                # Check if there's actual text content
                if not candidate.content.parts:
                    continue
                
                response_text = candidate.content.parts[0].text
                
                # Parse response
                addressees, confidence, reason = parse_gemini_response(
                    response_text, meeting_speakers
                )
                
                # Only proceed if confidence threshold met
                if addressees and confidence >= CONFIDENCE_THRESHOLD:
                    # Find if addressee responded
                    response_turn = None
                    response_idx = None
                    
                    for j in range(i+1, min(i+20, len(all_dacts))):
                        next_dact = all_dacts[j]
                        
                        if next_dact['speaker'] in addressees:
                            response_turn = next_dact
                            response_idx = j
                            break
                    
                    # Only create sequence if addressee responded
                    if response_turn:
                        # Get continuation
                        continuation = []
                        participants = set([dact['speaker']] + addressees)
                        
                        for k in range(response_idx + 1, len(all_dacts)):
                            next_dact = all_dacts[k]
                            
                            if next_dact['speaker'] in participants:
                                continuation.append(next_dact)
                            else:
                                if len(continuation) > 0:
                                    break
                        
                        # Create sequence
                        sequence = {
                            'meeting_id': meeting_id,
                            'sequence_id': f"{meeting_id}_inferred{len(sequences)}",
                            'context': [{'speaker': c['speaker'], 'text': c['text']} for c in context],
                            'addressing_turn': {
                                'speaker': dact['speaker'],
                                'addressees': addressees,
                                'text': dact['text'],
                                'is_explicit': False,
                                'inference_confidence': confidence,  # 0-10 scale
                                'inference_reason': reason
                            },
                            'response': {
                                'speaker': response_turn['speaker'],
                                'text': response_turn['text']
                            },
                            'continuation': [{'speaker': c['speaker'], 'text': c['text']} 
                                           for c in continuation]
                        }
                        
                        sequences.append(sequence)
                
                # Batch delay
                batch_count += 1
                if batch_count % BATCH_SIZE == 0:
                    time.sleep(API_DELAY)
                    print(f"    Processed {i+1}/{len(all_dacts)} turns, "
                          f"generated {len([s for s in sequences if s['meeting_id'] == meeting_id])} sequences...")
            
            except Exception as e:
                # This is normal - some turns may trigger safety filters or fail
                # Just skip them and continue
                if i % 100 == 0:  # Only print every 100 errors to avoid spam
                    print(f"    Note: Skipped some turns (safety filters or API issues)")
                continue
        
        print(f"  ✓ Generated {len([s for s in sequences if s['meeting_id'] == meeting_id])} sequences")
    
    # Cost estimation (Gemini API pricing: free tier has limits, paid is similar to Vertex)
    # With 10 turns of context
    avg_input_tokens = 650  # ~65 tokens per turn × 10 turns
    avg_output_tokens = 50  # Addressees + confidence + reasoning
    total_cost_estimate = (
        (total_api_calls * avg_input_tokens / 1000 * 0.00001875) +
        (total_api_calls * avg_output_tokens / 1000 * 0.000075)
    )
    
    print(f"\n" + "="*70)
    print(f"API Usage Summary:")
    print(f"  Total API calls: {total_api_calls:,}")
    print(f"  Estimated cost: ${total_cost_estimate:.4f}")
    print(f"  Note: Free tier includes 15 requests/min, 1M tokens/day")
    print("="*70)
    
    return sequences


# ============================================================================
# MERGE WITH EXPLICIT SEQUENCES
# ============================================================================

def merge_sequences(explicit_sequences: List[Dict], 
                   inferred_sequences: List[Dict]) -> List[Dict]:
    """Merge explicit and inferred sequences"""
    # Mark source for each sequence
    for seq in explicit_sequences:
        seq['source'] = 'explicit'
        # Explicit sequences now have confidence=10 and reason from Stage 1
        # No need to add None values anymore
    
    for seq in inferred_sequences:
        seq['source'] = 'gemini_inferred'
    
    all_sequences = explicit_sequences + inferred_sequences
    all_sequences.sort(key=lambda x: (x['meeting_id'], x['sequence_id']))
    
    return all_sequences


def print_statistics(explicit_seqs: List[Dict], inferred_seqs: List[Dict]):
    """Print comparison statistics"""
    print("\n" + "="*70)
    print("DATASET COMPARISON")
    print("="*70)
    
    print(f"\nExplicit sequences (Stage 1): {len(explicit_seqs):,}")
    print(f"Inferred sequences (Stage 1b): {len(inferred_seqs):,}")
    print(f"Total combined: {len(explicit_seqs) + len(inferred_seqs):,}")
    print(f"Dataset expansion: {len(inferred_seqs)/len(explicit_seqs)*100:.1f}% increase")
    
    if inferred_seqs:
        confidences = [s['addressing_turn']['inference_confidence'] for s in inferred_seqs]
        print(f"\nInferred sequence confidence (0-10 scale):")
        print(f"  Average: {sum(confidences)/len(confidences):.1f}/10")
        print(f"  Min: {min(confidences)}/10")
        print(f"  Max: {max(confidences)}/10")
        
        # Distribution
        from collections import Counter as ConfCounter
        conf_dist = ConfCounter(confidences)
        print(f"\n  Distribution:")
        for conf in sorted(conf_dist.keys(), reverse=True):
            count = conf_dist[conf]
            pct = count / len(confidences) * 100
            print(f"    {conf}/10: {count:,} ({pct:.1f}%)")
    
    explicit_meetings = Counter(s['meeting_id'] for s in explicit_seqs)
    inferred_meetings = Counter(s['meeting_id'] for s in inferred_seqs)
    
    print(f"\nSequences per meeting:")
    all_meetings = sorted(set(list(explicit_meetings.keys()) + list(inferred_meetings.keys())))
    for meeting in all_meetings:
        exp_count = explicit_meetings.get(meeting, 0)
        inf_count = inferred_meetings.get(meeting, 0)
        print(f"  {meeting}: {exp_count} explicit + {inf_count} inferred = {exp_count + inf_count} total")
    
    print("="*70)


def print_examples(inferred_seqs: List[Dict], num_examples: int = 3):
    """Print example inferred sequences"""
    print("\n" + "="*70)
    print("EXAMPLE INFERRED SEQUENCES")
    print("="*70 + "\n")
    
    for i, seq in enumerate(inferred_seqs[:num_examples]):
        print(f"Sequence {i+1}: {seq['sequence_id']}")
        print(f"Meeting: {seq['meeting_id']}")
        
        addr = seq['addressing_turn']
        print(f"Confidence: {addr['inference_confidence']}/10")
        print(f"Reasoning: {addr['inference_reason']}")
        
        print(f"\nContext ({len(seq['context'])} turns):")
        for turn in seq['context'][-2:]:
            print(f"  [{turn['speaker']}]: {turn['text'][:60]}...")
        
        print(f"\nAddressing Turn (INFERRED):")
        print(f"  [{addr['speaker']}] → {', '.join(addr['addressees'])}: {addr['text'][:80]}...")
        
        resp = seq['response']
        print(f"\nResponse:")
        print(f"  [{resp['speaker']}]: {resp['text'][:60]}...")
        
        print(f"\nContinuation: {len(seq['continuation'])} turns")
        print("\n" + "-"*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Stage 1b: Infer addressees with Gemini (Weak Supervision)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all meetings (full dataset)
  python stage1b_infer_addressees_apikey.py
  
  # Test with one meeting only
  python stage1b_infer_addressees_apikey.py --meeting ES2002a
  
  # Quick test: one meeting, first 50 turns
  python stage1b_infer_addressees_apikey.py --meeting ES2002a --max-turns 50
  
  # Test with first 100 turns per meeting (all meetings)
  python stage1b_infer_addressees_apikey.py --max-turns 100
        """
    )
    parser.add_argument(
        '--meeting',
        type=str,
        help='Process only this meeting ID (e.g., ES2002a). If not specified, processes all meetings.'
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        help='Maximum number of turns to process per meeting (for testing). If not specified, processes all turns.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_FILE,
        help=f'Output file for inferred sequences (default: {OUTPUT_FILE})'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Take every Nth high-quality dialogue act per meeting before Gemini (default: 1 = use all)'
    )
    parser.add_argument(
        '--max-samples-per-meeting',
        type=int,
        help='Optional cap on number of high-quality dialogue acts per meeting passed to Gemini'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("STAGE 1b: WEAK SUPERVISION WITH GEMINI (API KEY VERSION)")
    print("="*70)
    
    # Load explicit sequences
    print("\nLoading explicit sequences from Stage 1...")
    with open(EXPLICIT_SEQUENCES_FILE, 'r') as f:
        explicit_sequences = json.load(f)
    print(f"✓ Loaded {len(explicit_sequences):,} explicit sequences")
    
    # Get meeting IDs
    all_meeting_ids = sorted(set(s['meeting_id'] for s in explicit_sequences))
    
    # Filter to single meeting if specified
    if args.meeting:
        if args.meeting in all_meeting_ids:
            meeting_ids = [args.meeting]
            print(f"✓ Processing single meeting: {args.meeting}")
        else:
            print(f"\nERROR: Meeting '{args.meeting}' not found!")
            print(f"Available meetings: {', '.join(all_meeting_ids)}")
            return
    else:
        meeting_ids = all_meeting_ids
        print(f"✓ Found {len(meeting_ids)} meetings: {', '.join(meeting_ids)}")
    
    # Show test mode info
    if args.max_turns:
        print(f"✓ Test mode: Processing max {args.max_turns} turns per meeting")
    if args.stride and args.stride > 1:
        print(f"✓ Stride sampling: taking every {args.stride}th high-quality dialogue act per meeting")
    if args.max_samples_per_meeting:
        print(f"✓ Capping to max {args.max_samples_per_meeting} high-quality dialogue acts per meeting")
    
    # Initialize Gemini
    try:
        model, generation_config = init_gemini(API_KEY)
    except Exception as e:
        print(f"\nERROR: Failed to initialize Gemini API")
        print(f"  {e}")
        return
    
    # Infer addressees
    print("\n" + "="*70)
    print("INFERRING ADDRESSEES WITH GEMINI")
    print("="*70)
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Batch size: {BATCH_SIZE} (with {API_DELAY}s delay)")
    print(f"Max context turns: {MAX_CONTEXT_TURNS}")
    print()
    
    inferred_sequences = infer_sequences_with_gemini(
        AMI_CORPUS_DIR,
        meeting_ids,
        model,
        generation_config,
        max_turns_per_meeting=args.max_turns,
        stride=args.stride,
        max_samples_per_meeting=args.max_samples_per_meeting,
    )
    
    print(f"\n✓ Generated {len(inferred_sequences):,} inferred sequences")
    
    # Save outputs
    output_file = args.output
    with open(output_file, 'w') as f:
        json.dump(inferred_sequences, f, indent=2)
    print(f"✓ Saved inferred sequences to {output_file}")
    
    combined_sequences = merge_sequences(explicit_sequences, inferred_sequences)
    with open(COMBINED_OUTPUT, 'w') as f:
        json.dump(combined_sequences, f, indent=2)
    print(f"✓ Saved combined sequences to {COMBINED_OUTPUT}")
    
    # Statistics
    print_statistics(explicit_sequences, inferred_sequences)
    print_examples(inferred_sequences)
    
    print("\n" + "="*70)
    print("STAGE 1b COMPLETE")
    print("="*70)
    
    if args.meeting or args.max_turns:
        print("\n⚠️  TEST MODE - Processed limited data")
        if args.meeting:
            print(f"   Only processed meeting: {args.meeting}")
        if args.max_turns:
            print(f"   Limited to {args.max_turns} turns per meeting")
        print("\nTo process full dataset, run without flags:")
        print("   python stage1b_infer_addressees_apikey.py")
    else:
        print(f"\nNext step: Use '{COMBINED_OUTPUT}' as input for Stage 2")
        print("Or run Stage 2 with explicit sequences only (original pipeline)")


if __name__ == '__main__':
    main()
