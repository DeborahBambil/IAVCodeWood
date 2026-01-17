import os
import cv2

class BancoImagens:
    def __init__(self, nomeBancoImagens, pastaRaiz='./'):
        self.pastaRaiz = pastaRaiz
        self.pastaBancoImagens = os.path.join(self.pastaRaiz, nomeBancoImagens)
        self.nomeArquivoArff = os.path.join(self.pastaBancoImagens, nomeBancoImagens + ".arff")
        self.classes = self.lista_pastas()

    def lista_pastas(self):
        """Lista subpastas que representam as classes, ignorando pastas de sistema."""
        pastas_ignoradas = ['results', 'test', 'data', '__pycache__', '.ipynb_checkpoints']
        if not os.path.exists(self.pastaBancoImagens):
            return []
        
        return [name for name in os.listdir(self.pastaBancoImagens)
                if os.path.isdir(os.path.join(self.pastaBancoImagens, name)) 
                and name not in pastas_ignoradas]

    def imagens_da_classe(self, nomeDaClasse):
        """Retorna uma lista de matrizes de imagens lidas pelo OpenCV."""
        pastaDaClasse = os.path.join(self.pastaBancoImagens, nomeDaClasse)
        imagens = []
        extensoes_validas = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

        if not os.path.exists(pastaDaClasse):
            return []

        for item in os.listdir(pastaDaClasse):
            if item.lower().endswith(extensoes_validas):
                caminho = os.path.join(pastaDaClasse, item)
                try:
                    img = cv2.imread(caminho)
                    if img is not None and img.size > 0:
                        imagens.append(img)
                except:
                    continue
        return imagens