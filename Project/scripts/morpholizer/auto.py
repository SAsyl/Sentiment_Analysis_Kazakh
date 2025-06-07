import pyautogui
import keyboard
import time

totalDict_new = []


def main():

    words_small_example = ['аши', 'ари', 'ады', 'ашш', 'өзы']
    
    # Задержка, чтобы вы успели открыть сайт
    print("У вас есть 5 секунд, чтобы открыть нужный сайт...")
    time.sleep(5)
    
    # Начало автоматизации
    for word in words_small_example:
        # Клик по строке поиска (координаты зоны поиска на экране)
        pyautogui.click(x=1458, y=175)  # Замените координаты на актуальные для вашей строки поиска
    
        # Нажать "Ctrl + A", чтобы выделить текст
        keyboard.press_and_release("ctrl+a")
        time.sleep(0.1)
    
        # Нажать "Ctrl + V", чтобы вставить текст (установите слово в буфер обмена заранее)
        pyautogui.write(word)  # Или используйте pyperclip для копирования
        time.sleep(0.5)
    
        # Нажать Enter (поиск)
        keyboard.press_and_release("enter")
        time.sleep(2)  # Подождать, пока данные появятся на экране
    
        # Клик по нужной зоне, чтобы выделить информацию
        pyautogui.doubleClick(x=612, y=392)  # Координаты зоны с текстом
        time.sleep(0.5)
    
        # Скопировать выделенный текст (Ctrl + C)
        keyboard.press_and_release("ctrl+c")
        time.sleep(0.5)
    
        # Вставить скопированный текст в input() для обработки
        input_words = pyautogui.paste()  # Или используйте библиотеку pyperclip
        print(f"Скопированный текст: {input_words}")
    
        if inputWords.startswith('В словаре имеются схожие по написанию слова:'):
            inputWords = inputWords.split(': ')[-1]
                
        inputWords = inputWords.split(', ')
        
        totalDict_new = totalDict_new + inputWords
    
        # Добавьте паузу перед обработкой следующего слова
        time.sleep(1)
    
    print("Готово!")

if __name__ == '__main__':
    main()

    print(totalDict_new)
