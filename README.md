# LLM with RAG: Retrieval-Augmented Generation with Response Caching 🚀

Реализация продвинутой системы вопрос-ответ с использованием метода RAG (Retrieval-Augmented Generation) 
и интеллектуального кеширования ответов для оптимизации работы языковых моделей.

## 🔍 Основные особенности

- **🦾 Гибридная архитектура RAG**  
  Комбинация нейросетевых моделей с поиском по знаниям для точных ответов

- **⚡ Система кеширования**  
  Многоуровневое кеширование результатов для:
  - Уменьшения времени ответа на повторяющиеся запросы
  - Снижения затрат на вызовы LLM
  - Оптимизации использования вычислительных ресурсов

- **📚 Поддержка форматов данных**  
  Работа с  PDF
  В дальнейшем расшириться список поддерживаемых форматов 

- **🧠 Интеграция моделей**  
  Поддержка популярных LLM (Qwen). Так как система задумывалась работать локально, то подойлет любая модель,
  которую сможет поддерживать железо
