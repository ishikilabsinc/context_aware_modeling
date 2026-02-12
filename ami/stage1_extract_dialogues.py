#!/usr/bin/env python3
"""
Stage 1: Extract Dialogue Sequences from AMI Corpus

Goal:
    Parse AMI XML files and extract time-ordered dialogue sequences with:
    - Context turns (before addressing)
    - Addressing turn (with explicit addressee)
    - Response turn (from addressee)
    - Continuation turns (follow-up conversation)

Input:
    - AMI Corpus XML files (dialogue acts, words, meetings)

Output:
    - stage1_sequences.json - conversation sequences for Stage 2
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import json
import re
from typing import List, Dict, Optional
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Root directory of the AMI corpus relative to this repo.
# This makes the script robust to moving the repository.
AMI_CORPUS_DIR = str(Path(__file__).resolve().parents[1] / 'datasets' / 'ami_public_manual_1.6.2')

# Directory for all intermediate JSON dumps across stages
JSON_DUMPS_DIR = Path(__file__).parent / 'json_dumps'
JSON_DUMPS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = JSON_DUMPS_DIR / 'stage1_sequences.json'

def load_meeting_speakers(corpus_dir: str) -> Dict:
    """
    Load speaker mappings from meetings.xml
    
    Returns:
        Dict mapping meeting_id -> speaker_letter -> {participant_id, role}
    """
    corpus_path = Path(corpus_dir)
    meetings_file = corpus_path / 'corpusResources' / 'meetings.xml'
    
    if not meetings_file.exists():
        print(f"Warning: {meetings_file} not found")
        return {}
    
    tree = ET.parse(meetings_file)
    root = tree.getroot()
    ns = {'nite': 'http://nite.sourceforge.net/'}
    
    meeting_speakers = {}
    
    for meeting in root.findall('.//meeting', ns):
        meeting_id = meeting.get('observation')
        
        if not meeting_id:
            continue
        
        speakers = {}
        for speaker in meeting.findall('.//speaker', ns):
            speaker_letter = speaker.get('nxt_agent')
            participant_id = speaker.get('global_name')
            role = speaker.get('role')
            
            if speaker_letter:
                speakers[speaker_letter] = {
                    'participant_id': participant_id,
                    'role': role,
                }
        
        if speakers:
            meeting_speakers[meeting_id] = speakers
    
    return meeting_speakers


def extract_text_from_words_xml(words_file: Path) -> Dict[str, Dict]:
    """
    Extract text content and timing from AMI words XML file
    
    Args:
        words_file: Path to the words XML file
        
    Returns:
        Dict mapping word_id -> {text, starttime, endtime}
    """
    if not words_file.exists():
        return {}
    
    tree = ET.parse(words_file)
    root = tree.getroot()
    ns = {'nite': 'http://nite.sourceforge.net/'}
    
    word_data = {}
    for word in root.findall('.//w', ns):
        word_id = word.get('{http://nite.sourceforge.net/}id', '')
        text = word.text if word.text else ''
        starttime = word.get('starttime', '')
        endtime = word.get('endtime', '')
        
        word_data[word_id] = {
            'text': text,
            'starttime': float(starttime) if starttime else 0.0,
            'endtime': float(endtime) if endtime else 0.0,
        }
    
    return word_data


def parse_word_reference(ref: str, word_data: Dict[str, Dict]) -> List[str]:
    """
    Parse word reference from dialogue act, handling ranges
    
    Example: "id(word1)..id(word10)" -> ['word1', 'word2', ..., 'word10']
    """
    pattern = r'id\(([^)]+)\)'
    word_ids = re.findall(pattern, ref)
    
    # Handle range references (e.g., id(word1)..id(word10))
    if len(word_ids) == 2 and '..' in ref:
        start_id = word_ids[0]
        end_id = word_ids[1]
        
        start_match = re.search(r'(.+\.words)(\d+)$', start_id)
        end_match = re.search(r'(.+\.words)(\d+)$', end_id)
        
        if start_match and end_match:
            base = start_match.group(1)
            start_num = int(start_match.group(2))
            end_num = int(end_match.group(2))
            
            all_ids = []
            for num in range(start_num, end_num + 1):
                word_id = f"{base}{num}"
                if word_id in word_data:
                    all_ids.append(word_id)
            return all_ids
    
    return word_ids


def extract_dialogue_act_text_and_time(dact_elem, words_dir: Path, meeting_id: str, speaker: str) -> Optional[Dict]:
    """
    Extract text content and timing for a dialogue act
    
    Resolves word references from dialogue act XML to actual text from words XML
    """
    ns = {'nite': 'http://nite.sourceforge.net/'}
    
    children = dact_elem.findall('.//nite:child', ns)
    if not children:
        return None
    
    words_file = words_dir / f'{meeting_id}.{speaker}.words.xml'
    word_data = extract_text_from_words_xml(words_file)
    
    all_words = []
    all_starttimes = []
    all_endtimes = []
    
    for child in children:
        href = child.get('href', '')
        if not href:
            continue
        
        word_ids = parse_word_reference(href, word_data)
        
        for word_id in word_ids:
            # Try with full meeting prefix
            full_word_id = f'{meeting_id}.{speaker}.words{word_id.split("words")[-1]}' if 'words' in word_id else word_id
            
            if full_word_id in word_data:
                word_info = word_data[full_word_id]
                all_words.append(word_info['text'])
                if word_info['starttime']:
                    all_starttimes.append(word_info['starttime'])
                if word_info['endtime']:
                    all_endtimes.append(word_info['endtime'])
            elif word_id in word_data:
                word_info = word_data[word_id]
                all_words.append(word_info['text'])
                if word_info['starttime']:
                    all_starttimes.append(word_info['starttime'])
                if word_info['endtime']:
                    all_endtimes.append(word_info['endtime'])
    
    if not all_words:
        return None
    
    return {
        'text': ' '.join(all_words).strip(),
        'starttime': min(all_starttimes) if all_starttimes else 0.0,
        'endtime': max(all_endtimes) if all_endtimes else 0.0,
    }


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_conversation_sequences_ami(corpus_dir: str, meeting_ids: List[str] = None) -> List[Dict]:
    """
    Extract conversation sequences from AMI corpus with explicit addressing
    
    Process:
        1. Auto-discover meetings with addressee annotations (if meeting_ids not provided)
        2. For each meeting, parse all dialogue acts
        3. Find addressing events where:
           - Someone explicitly addresses another person (addressee attribute)
           - The addressee actually responds (within next 20 turns)
        4. Package as: context → addressing → response → continuation
    
    Args:
        corpus_dir: Path to AMI corpus root
        meeting_ids: Optional list of specific meeting IDs to process
        
    Returns:
        List of conversation sequences
    """
    corpus_path = Path(corpus_dir)
    dialogue_acts_dir = corpus_path / 'dialogueActs'
    words_dir = corpus_path / 'words'
    
    # Auto-discover meetings with addressee annotations if not specified
    if meeting_ids is None:
        meeting_ids = []
        for da_file in dialogue_acts_dir.glob('*.dialog-act.xml'):
            meeting_id = da_file.stem.split('.')[0]
            if meeting_id not in meeting_ids:
                tree = ET.parse(da_file)
                ns = {'nite': 'http://nite.sourceforge.net/'}
                # Only include meetings with addressee annotations
                if tree.findall('.//dact[@addressee]', ns):
                    meeting_ids.append(meeting_id)
        meeting_ids = sorted(set(meeting_ids))[:10]  # Limit to first 10 meetings
        print(f"Auto-discovered {len(meeting_ids)} meetings with addressee annotations")
    
    sequences = []
    
    for meeting_id in meeting_ids:
        print(f"Processing {meeting_id}...")
        
        # Collect all dialogue acts for this meeting
        all_dacts = []
        
        for da_file in dialogue_acts_dir.glob(f'{meeting_id}.*.dialog-act.xml'):
            speaker = da_file.stem.split('.')[1]
            
            tree = ET.parse(da_file)
            ns = {'nite': 'http://nite.sourceforge.net/'}
            
            for dact in tree.findall('.//dact', ns):
                addressee = dact.get('addressee', '').strip()
                dact_data = extract_dialogue_act_text_and_time(dact, words_dir, meeting_id, speaker)
                
                if dact_data and dact_data['text']:
                    all_dacts.append({
                        'meeting_id': meeting_id,
                        'speaker': speaker,
                        'addressee': addressee.split(',') if addressee else [],
                        'text': dact_data['text'],
                        'starttime': dact_data['starttime'],
                        'endtime': dact_data['endtime'],
                        'dact_id': dact.get('{http://nite.sourceforge.net/}id', ''),
                        'has_explicit_addressee': bool(addressee),
                    })
        
        # Sort by time
        all_dacts.sort(key=lambda x: (x['starttime'], x['speaker']))
        
        # Extract sequences where someone is addressed and responds
        i = 0
        while i < len(all_dacts):
            dact = all_dacts[i]
            
            # Look for explicit addressing events
            if dact['has_explicit_addressee']:
                addressees = dact['addressee']
                addressing_speaker = dact['speaker']
                
                # Get context (up to 10 turns before for better context understanding)
                context = all_dacts[max(0, i-10):i]
                
                # Find response from one of the addressees (within next 20 turns)
                response = None
                response_idx = None
                
                for j in range(i+1, min(i+20, len(all_dacts))):
                    next_dact = all_dacts[j]
                    
                    if next_dact['speaker'] in addressees:
                        response = next_dact
                        response_idx = j
                        break
                
                # Only keep sequences where addressee actually responded
                if response:
                    # Get continuation (ongoing exchange between participants)
                    continuation = []
                    participants = set([addressing_speaker] + addressees)
                    
                    for k in range(response_idx + 1, len(all_dacts)):
                        next_dact = all_dacts[k]
                        
                        # Stop if someone addresses a different set of people
                        if next_dact['has_explicit_addressee']:
                            new_addressees = set(next_dact['addressee'])
                            current_addressees = set(addressees)
                            
                            if new_addressees != current_addressees:
                                break
                        
                        # Include if speaker is in the participant set
                        if next_dact['speaker'] in participants:
                            continuation.append(next_dact)
                        else:
                            # Stop if someone outside the exchange speaks
                            if len(continuation) > 0:
                                break
                    
                    # Create sequence
                    sequences.append({
                        'meeting_id': meeting_id,
                        'sequence_id': f"{meeting_id}_seq{len(sequences)}",
                        'context': [{'speaker': c['speaker'], 'text': c['text']} for c in context],
                        'addressing_turn': {
                            'speaker': addressing_speaker,
                            'addressees': addressees,
                            'text': dact['text'],
                            'is_explicit': True,
                            'inference_confidence': 10,  # Ground truth from corpus annotations
                            'inference_reason': 'Explicit'
                        },
                        'response': {
                            'speaker': response['speaker'],
                            'text': response['text']
                        },
                        'continuation': [{'speaker': c['speaker'], 'text': c['text']} for c in continuation]
                    })
                    
                    # Skip ahead past this sequence
                    i = response_idx + len(continuation) + 1
                else:
                    i += 1
            else:
                i += 1
        
        print(f"  Extracted {len([s for s in sequences if s['meeting_id'] == meeting_id])} sequences from {meeting_id}")
    
    return sequences


def print_statistics(sequences: List[Dict]):
    """Print statistics about extracted sequences"""
    print("\n" + "="*70)
    print("STAGE 1 STATISTICS")
    print("="*70)
    
    # Meetings
    meetings = [s['meeting_id'] for s in sequences]
    meeting_counts = Counter(meetings)
    print(f"\nMeetings processed: {len(meeting_counts)}")
    for meeting, count in meeting_counts.most_common():
        print(f"  {meeting}: {count} sequences")
    
    # Speakers
    all_speakers = set()
    for seq in sequences:
        all_speakers.add(seq['addressing_turn']['speaker'])
        all_speakers.update(seq['addressing_turn']['addressees'])
        all_speakers.add(seq['response']['speaker'])
    print(f"\nUnique speakers: {len(all_speakers)}")
    print(f"  {', '.join(sorted(all_speakers))}")
    
    # Sequence lengths
    context_lengths = [len(s['context']) for s in sequences]
    continuation_lengths = [len(s['continuation']) for s in sequences]
    
    print(f"\nContext turns:")
    print(f"  Average: {sum(context_lengths)/len(context_lengths):.1f}")
    print(f"  Range: {min(context_lengths)} - {max(context_lengths)}")
    
    print(f"\nContinuation turns:")
    print(f"  Average: {sum(continuation_lengths)/len(continuation_lengths):.1f}")
    print(f"  Range: {min(continuation_lengths)} - {max(continuation_lengths)}")
    
    print(f"\n" + "="*70)


def print_examples(sequences: List[Dict], num_examples: int = 3):
    """Print example sequences"""
    print("\n" + "="*70)
    print("EXAMPLE SEQUENCES")
    print("="*70 + "\n")
    
    for i, seq in enumerate(sequences[:num_examples]):
        print(f"Sequence {i+1}: {seq['sequence_id']}")
        print(f"Meeting: {seq['meeting_id']}")
        print(f"\nContext ({len(seq['context'])} turns):")
        for turn in seq['context']:
            print(f"  [{turn['speaker']}]: {turn['text'][:80]}...")
        
        addr = seq['addressing_turn']
        print(f"\nAddressing Turn:")
        print(f"  [{addr['speaker']}] → {', '.join(addr['addressees'])}: {addr['text'][:80]}...")
        
        resp = seq['response']
        print(f"\nResponse:")
        print(f"  [{resp['speaker']}]: {resp['text'][:80]}...")
        
        print(f"\nContinuation: {len(seq['continuation'])} turns")
        for turn in seq['continuation'][:2]:
            print(f"  [{turn['speaker']}]: {turn['text'][:80]}...")
        
        print("\n" + "-"*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("Starting AMI sequence extraction...")
    print("="*70)
    
    # Load meeting metadata
    meeting_speakers = load_meeting_speakers(AMI_CORPUS_DIR)
    print(f"Loaded {len(meeting_speakers)} meetings with speaker information")
    
    # Extract sequences
    sequences = extract_conversation_sequences_ami(AMI_CORPUS_DIR)
    
    print("="*70)
    print(f"\nTotal sequences extracted: {len(sequences)}")
    
    # Print examples and statistics
    print_examples(sequences)
    print_statistics(sequences)
    
    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(sequences, f, indent=2)
    
    print(f"\n✓ Saved {len(sequences)} sequences to {OUTPUT_FILE}")
    print(f"\nReady for Stage 2: Generate Decision Points")


if __name__ == '__main__':
    main()
