import re
import os
import json
import logging
import difflib
from datetime import datetime
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

# Настройка JSON логгирования
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": "corrector",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Добавляем дополнительные поля если есть
        if hasattr(record, 'task_id'):
            log_entry["task_id"] = record.task_id
        if hasattr(record, 'text_length'):
            log_entry["text_length"] = record.text_length
        if hasattr(record, 'processing_time'):
            log_entry["processing_time"] = record.processing_time
        if hasattr(record, 'corrections_count'):
            log_entry["corrections_count"] = record.corrections_count
        if hasattr(record, 'method'):
            log_entry["method"] = record.method
        if hasattr(record, 'device'):
            log_entry["device"] = record.device
        
        return json.dumps(log_entry, ensure_ascii=False)

# Настройка логгера
logger = logging.getLogger("corrector")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

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
        logger.info("Инициализация TextCorrector", extra={
            'device': str(self.device),
            'model_name': self.MODEL_NAME
        })
        
        # Перемещаем T5 модель на device
        self.model = self.model.to(self.device)
        logger.info("T5 модель перемещена на device", extra={'device': str(self.device)})
        
        # Загружаем Silero модель
        logger.info("Загрузка Silero модели")
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
        logger.info("Silero модель успешно загружена")

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
        
        import time
        start_time = time.time()
        
        logger.info("Начало обработки Silero Text Enhancement", extra={
            'method': 'enhance_text',
            'text_length': len(text)
        })
        
        # Убираем всю пунктуацию и переводим в нижний регистр
        import re
        cleaned_text = re.sub(r'[^\w\s]', ' ', text).lower()
        # Убираем лишние пробелы
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        logger.info("Текст очищен для Silero TE", extra={
            'method': 'enhance_text',
            'original_length': len(text),
            'cleaned_length': len(cleaned_text)
        })
        
        try:
            # Применяем Silero Text Enhancement к очищенному тексту
            enhanced = self.apply_te(cleaned_text, lan="ru")
            
            processing_time = time.time() - start_time
            
            logger.info("Silero Text Enhancement успешно завершен", extra={
                'method': 'enhance_text',
                'processing_time': processing_time,
                'enhanced_length': len(enhanced)
            })
            
            return enhanced
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error("Ошибка в enhance_text", extra={
                'method': 'enhance_text',
                'error': str(e),
                'processing_time': processing_time
            })
            
            return text

    def correct(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text or not text.strip():
            return text, {}
        
        import time
        start_time = time.time()
        
        logger.info("Начало полной коррекции текста", extra={
            'method': 'correct',
            'text_length': len(text)
        })
        
        # Нейронная коррекция орфографии
        neural_start = time.time()
        neural_corrected = self.neural_spell_correct_long(text)
        neural_time = time.time() - neural_start
        
        logger.info("Нейронная коррекция завершена", extra={
            'method': 'correct',
            'step': 'neural_spell_correct',
            'processing_time': neural_time,
            'corrected_length': len(neural_corrected)
        })
        
        # Улучшение текста через Silero TE
        enhance_start = time.time()
        enhanced_text = self.enhance_text(neural_corrected)
        enhance_time = time.time() - enhance_start
        
        logger.info("Text Enhancement завершен", extra={
            'method': 'correct',
            'step': 'enhance_text',
            'processing_time': enhance_time,
            'enhanced_length': len(enhanced_text)
        })
        
        # Коррекция терминов
        terms_start = time.time()
        cleaned_text, corrections = self.correct_terms_preserve_structure(enhanced_text, self.correct_words)
        terms_time = time.time() - terms_start
        
        total_time = time.time() - start_time
        
        logger.info("Коррекция терминов завершена", extra={
            'method': 'correct',
            'step': 'correct_terms',
            'processing_time': terms_time,
            'corrections_count': len(corrections),
            'final_length': len(cleaned_text)
        })
        
        logger.info("Полная коррекция текста завершена", extra={
            'method': 'correct',
            'total_processing_time': total_time,
            'neural_time': neural_time,
            'enhance_time': enhance_time,
            'terms_time': terms_time,
            'corrections_count': len(corrections),
            'final_length': len(cleaned_text)
        })
        
        return cleaned_text, corrections
