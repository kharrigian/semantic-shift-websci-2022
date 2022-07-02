
"""
Keyword/phrase Matching

Example Usage:
```
## Imports
from pprint import pformat
from semshift.model.track import Tracker

## Define Terms to Match
term_list = [
    "mental health",
    "not depres*",
    "*happ*",
    "#COVID",
    "depres*",
]

## Examples
examples = [
    "I am not feeling happy today. I think my mental health is suffering :/",
    "Definitely not depressed! But not happy either. Some might say I'm #unhappy",
    "#COVID has not been good for my MeNtaL HeaLtH",
    "@DepressionMemes You are my favority account!"
]

## Initialize Tracker and Add Terms
tracker = Tracker(include_mentions=False)
tracker = tracker.add_terms(term_list, include_hashtags=True)

## Search
matches = []
for example in examples:
    example_matches = tracker.search(example)
    matches.append(example_matches)

## Show Results
print(pformat(matches))
```
"""

######################
### Imports
######################

## Standard Library
import re
import string

## Project Tools
from ..util.logging import initialize_logger

######################
### Globals
######################

## Logger
LOGGER = initialize_logger()

## Special Characters
PUNCTUATION = string.punctuation.replace("#","")
SPECIAL = "“”…‘’´"

######################
### Tracker Class
######################

class Tracker(object):

    """

    """

    def __init__(self,
                 include_mentions=False):
        """

        """
        ## Properties
        self._include_mentions = include_mentions
        ## Term Set
        self._terms = {}
    
    def _create_regex_dict(self,
                           terms,
                           include_hashtag=True):
        """
        
        """
        ## Initialize Cache
        regex_dict = {}
        ## Add Terms to Dictionary
        for t in terms:
            ## Stem Type
            is_prefix_stem = t.endswith("*") ## String Starts With Term
            is_suffix_stem = t.startswith("*") ## String Ends With Term
            is_hashtag = t.startswith("#") ## String is a Hashtag
            is_multi_gram = t.count(" ") > 0
            ## Format Root
            t_stem = re.escape(t.rstrip("*").lstrip("*"))
            ## Parse Term
            if t_stem.isupper():
                regex_dict[t] = (re.compile(t_stem), is_prefix_stem, is_suffix_stem, is_hashtag)
                if include_hashtag and not is_hashtag and not is_multi_gram:
                    regex_dict["#"+t] = (re.compile("#" + t_stem), is_prefix_stem, is_suffix_stem, False)
            else:
                regex_dict[t] = (re.compile(t_stem, re.IGNORECASE), is_prefix_stem, is_suffix_stem, is_hashtag)
                if include_hashtag and not is_hashtag and not is_multi_gram:
                    regex_dict["#"+t] = (re.compile("#"+t_stem, re.IGNORECASE), is_prefix_stem, is_suffix_stem, False)
        return regex_dict
    
    def _starts_with(self,
                     text,
                     index,
                     prefix):
        """

        """
        i = index
        while i >= 0:
            if text[i] == " ":
                return False
            if text[i] == prefix:
                return True
            i -= 1
        return False

    def _search_full_words(self,
                           text,
                           regex,
                           is_prefix_stem=False,
                           is_suffix_stem=False,
                           include_mentions=False):
        """
        
        """
        matches = []
        L = len(text)
        for match in regex.finditer(text):
            ## Get Span
            match_span = match.span()
            ## See if the Span Lies at the Start of the Text
            if include_mentions:
                starts_text = match_span[0] == 0 or text[match_span[0]-1] == " " or text[match_span[0]-1] in (PUNCTUATION + SPECIAL)
            else:
                starts_text = match_span[0] == 0 or text[match_span[0]-1] == " " or (text[match_span[0]-1] in (PUNCTUATION + SPECIAL) and not self._starts_with(text,match_span[0],"@"))
            ## See if the Span Lies at the End of the Text
            ends_text = match_span[1] == L or text[match_span[1]] == " " or text[match_span[1]] in (PUNCTUATION + SPECIAL)
            ## Determine Validity
            valid = True
            if not starts_text and not is_suffix_stem:
                valid = False
            if not ends_text and not is_prefix_stem:
                valid = False
            ## Cache
            if valid:
                match_start, match_end = match_span
                while match_start > 0 and len(text[match_start-1].strip()) == 1:
                    match_start -= 1
                while match_end < len(text) and len(text[match_end].strip()) == 1:
                    match_end += 1
                matches.append((text[match_start:match_end], match_span))
        return matches

    def _pattern_match(self,
                       text,
                       pattern_re,
                       include_mentions=False):
        """

        """
        ## Initialize Catch
        matches = []
        ## Search Through Patterns
        for keyword, (pattern, is_prefix_stem, is_suffix_stem, _) in pattern_re.items():
            ## Look for Matches
            keyword_matches = self._search_full_words(text,
                                                      pattern,
                                                      is_prefix_stem=is_prefix_stem,
                                                      is_suffix_stem=is_suffix_stem,
                                                      include_mentions=include_mentions)
            ## Format Matches
            keyword_matches = [(keyword, k[0], k[1]) for k in keyword_matches]
            ## Cache Matches
            matches.extend(keyword_matches)
        return matches
    
    def add_terms(self,
                  terms,
                  include_hashtags=False):
        """

        """
        _terms = self._create_regex_dict(terms, include_hashtags)
        self._terms.update(_terms)
        return self
    
    def search(self,
               text):
        """

        """
        ## Get Matches
        matches = self._pattern_match(text,
                                      self._terms,
                                      self._include_mentions)
        return matches
    

