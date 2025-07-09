import re
import difflib
from typing import Tuple, Dict, Optional
from natasha import MorphVocab
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import torch
import Levenshtein
import nltk

nltk.download('punkt', quiet=True, download_dir='/root/nltk_data')
nltk.download('punkt_tab', quiet=True, download_dir='/root/nltk_data')
from nltk.tokenize import sent_tokenize

morph = MorphVocab()

class TextCorrector:
    def __init__(self):
        self.MODEL_NAME = 'UrukHan/t5-russian-spell'
        self.tokenizer = T5TokenizerFast.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)

        self.correct_words = {
            "гендальф", "алладин", "старбакс", "найк", "тесла", "майкрософт", "якитория",
            "1С", "бизнес", "прокачка", "сопровождение", "внедрение", "маркетинг",
            "фреш", "IT", "ЮФО", "ERP", "УНФ", "CRM", "Розница", "Лицензии",
            "Битрикс", "Стахановец", "УСН", "РСВ", "НДС", "6-НДФЛ", "3-НДФЛ",
            "4-ФСС", "ИТС", "консалтинг", "SEO", "верстка", "Wix", "Tilda",
            "OpenCart", "API", "Энтерпрайз", "WordPress", "аутсорсинг", "госсектор",
            "ЭДО", "ЭЦП", "СПАРК", "Фреш", "Контрагент", "Коннект", "Линк", "РПД", "UMI",
            "маркетплейс"
        }

        self.typos = {
            "гендальс": "гендальф",
            "гендольф": "гендальф",
            "хендальф": "гендальф",
            "\"гендальс\",": "гендальф,",
            "\"гендольф\"": "гендальф",
            "\"хендальф\".": "гендальф.",
            "компния": "компания",
            "напровление": "направление",
            "сопрождения": "сопровождения",
            "якиротия,": "якитория",
            "унф": "УНФ",
            "tilda,": "Tilda",
            "токже": "также",
            "настрайкой": "настройкой",
            "тесло": "тесла",
            "несмотря": "«Несмотря",
            "отметоть,": "отметить,",
            "запск": "запуск",
            "маркетплейса": "маркетплейса",
        }

        self.stopwords = {
            "и", "в", "на", "по", "не", "что", "это", "было", "была",
            "он", "она", "они", "мы", "вы", "я", "его", "их", "её",
            "компания", "фирма", "инвестор", "проект", "развитие", "решение",
            "технологии", "данные", "год", "мнению", "аналитиков", "рынке",
            "присутствие", "продуктов", "питания", "сотрудничество", "стало",
            "пришел", "зонт", "посчитала", "перестал", "связавшись",
            "домой", "дверь", "дождь", "забыл", "возможно", "предложили", "аутсорсинг",
            "сервисами", "верстка", "услуг", "партнерство","верстку", "гендальф", "алладин", "старбакс", "найк", "тесла", "майкрософт", "якитория",
            "1С", "бизнес", "прокачка", "сопровождение", "внедрение", "маркетинг",
            "фреш", "IT", "ЮФО", "ERP", "УНФ", "CRM", "Розница", "Лицензии",
            "Битрикс", "Стахановец", "УСН", "РСВ", "НДС", "6-НДФЛ", "3-НДФЛ",
            "4-ФСС", "ИТС", "консалтинг", "SEO", "верстка", "Wix", "Tilda",
            "OpenCart", "API", "Энтерпрайз", "WordPress", "аутсорсинг", "госсектор",
            "ЭДО", "ЭЦП", "СПАРК", "Фреш", "Контрагент", "Коннект", "Линк", "РПД", "UMI",
            "маркетплейс"
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется device: {self.device}")
        
        # Перемещаем T5 модель на device
        self.model = self.model.to(self.device)
        
        # Загружаем Silero модель
        (
            self.silero_model,
            self.example_texts,
            self.languages,
            self.punct,
            self.apply_te
        ) = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_te',
            trust_repo=True
        )
        
        # Перемещаем Silero модель на device
        # self.silero_model = self.silero_model.to(self.device)

    def clean_word(self, word: str) -> str:
        return re.sub(r'[^\w\s]', '', word).lower()

    def normalize_word(self, word: str) -> str:
        parsed = morph.parse(word)
        if parsed:
            return parsed[0].normal
        return word

    def correct_word(
        self,
        word: str,
        correct_set: set,
        threshold_ratio: float = 0.7,
        min_similarity: float = 0.75
    ) -> Tuple[str, Optional[str]]:
        cleaned_word = self.clean_word(word)
        if len(cleaned_word) < 4 or cleaned_word in self.stopwords:
            return word, None
        normalized_word = self.normalize_word(cleaned_word)
        if normalized_word in correct_set:
            return word, None
        best_match = None
        min_distance = float('inf')
        for correct_word_ in correct_set:
            dist = Levenshtein.distance(normalized_word, correct_word_)
            max_len = max(len(normalized_word), len(correct_word_))
            similarity_ratio = 1 - (dist / max_len)
            if similarity_ratio > threshold_ratio and similarity_ratio >= min_similarity and dist < min_distance:
                min_distance = dist
                best_match = correct_word_
        if best_match:
            return best_match, word
        else:
            return word, None

    def tokenize_with_spaces(self, text: str):
        return re.findall(r'\s+|\w+|[^\w\s]', text, re.UNICODE)

    def correct_terms_preserve_structure(self, text: str, correct_set: set) -> Tuple[str, Dict[str, str]]:
        tokens = self.tokenize_with_spaces(text)
        corrections = {}
        corrected_tokens = []
        for token in tokens:
            try:
                if token.strip() == '' or not re.match(r'\w+', token):
                    corrected_tokens.append(token)
                else:
                    corrected, original = self.correct_word(token, correct_set)
                    corrected_tokens.append(corrected)
                    if original:
                        corrections[original] = corrected
            except Exception as e:
                print(f"Ошибка в correct_terms_preserve_structure при токене: '{token}'. Ошибка: {e}")
                corrected_tokens.append(token)
        corrected_text = ''.join(corrected_tokens)
        return corrected_text, corrections

    def normalize_for_metrics(self, word: str) -> str:
        return re.sub(r'[^\w\s]', '', word).lower()

    def find_real_corrections(self, orig_text: str, corrected_text: str) -> Dict[str, str]:
        orig_words = orig_text.split()
        corr_words = corrected_text.split()
        sm = difflib.SequenceMatcher(None, orig_words, corr_words)
        corrections = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace':
                for o_word, c_word in zip(orig_words[i1:i2], corr_words[j1:j2]):
                    if o_word != c_word:
                        corrections[o_word] = c_word
        return corrections

    def calculate_precision(self, corrections: dict, typos: dict) -> float:
        normalized_typos = {self.normalize_for_metrics(k): v.lower() for k, v in typos.items()}
        tp = 0
        fp = 0
        for orig, corr in corrections.items():
            norm_orig = self.normalize_for_metrics(orig)
            norm_corr = corr.lower()
            if norm_orig in normalized_typos:
                if norm_corr == normalized_typos[norm_orig]:
                    tp += 1
                else:
                    fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        return precision

    def neural_spell_correct_long(self, text: str) -> str:
        if not text or not text.strip():
            return text
        sentences = sent_tokenize(text, language='russian')
        if not sentences:
            return text
        corrected_sentences = []
        for sent in sentences:
            if not sent.strip():
                continue
            try:
                encoded = self.tokenizer(
                    "Spell correct: " + sent,
                    padding="longest",
                    max_length=256,
                    truncation=True,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoded.input_ids.to(self.device),
                        attention_mask=encoded.attention_mask.to(self.device),
                        max_length=256
                    )
                if len(outputs) == 0 or outputs[0].size(0) == 0:
                    corrected = sent
                else:
                    corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Ошибка в neural_spell_correct_long при обработке предложения: '{sent}'. Ошибка: {e}")
                corrected = sent
            corrected_sentences.append(corrected)
        return " ".join(corrected_sentences)

    def enhance_text(self, text: str) -> str:
        if not text or not text.strip():
            return text
        try:
            return self.apply_te(text, lan='ru')
        except Exception as e:
            print(f"Ошибка в enhance_text при обработке текста: '{text}'. Ошибка: {e}")
            return text

    def correct(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text or not text.strip():
            return text, {}
        neural_corrected = self.neural_spell_correct_long(text)
        enhanced_text = self.enhance_text(neural_corrected)
        cleaned_text, corrections = self.correct_terms_preserve_structure(enhanced_text, self.correct_words)
        return cleaned_text, corrections
