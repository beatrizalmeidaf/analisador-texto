// Gêneros textuais da versão original (classla/xlm-roberta-base-multilingual-text-genre-classifier).
// No navegador a classificação é zero-shot: comparamos o embedding médio do texto
// com o embedding de frases-protótipo de cada gênero.
export interface GenreDef {
  label: string;
  description: string;
  prototypes: string[];
}

export const GENRES: GenreDef[] = [
  {
    label: 'Informativo/Explicativo',
    description: 'Transmite conhecimento ou explica conceitos, processos e fenômenos com linguagem objetiva.',
    prototypes: [
      'Este texto explica um conceito científico e descreve como funciona um processo.',
      'Artigo acadêmico ou didático que apresenta definições, métodos e resultados de pesquisa.',
      'Texto enciclopédico que descreve características, história e classificação de um tema.',
    ],
  },
  {
    label: 'Notícia',
    description: 'Relata fatos recentes de interesse público de forma objetiva, respondendo o quê, quem, quando e onde.',
    prototypes: [
      'Notícia jornalística: nesta segunda-feira, o governo anunciou novas medidas, segundo o ministério.',
      'Reportagem sobre um acontecimento recente, com declarações de autoridades e testemunhas.',
    ],
  },
  {
    label: 'Instrução',
    description: 'Ensina procedimentos: manuais, tutoriais, receitas e guias passo a passo.',
    prototypes: [
      'Passo a passo: primeiro abra o menu, depois clique em configurações e selecione a opção.',
      'Receita: misture os ingredientes, leve ao forno por 30 minutos e sirva quente.',
      'Manual de instruções com etapas para instalar e configurar o produto.',
    ],
  },
  {
    label: 'Opinião/Argumentação',
    description: 'Defende um ponto de vista com argumentos: artigos de opinião, editoriais, resenhas e ensaios.',
    prototypes: [
      'Na minha opinião, essa decisão é um erro, e defendo que precisamos mudar de rumo.',
      'Artigo de opinião que argumenta a favor de uma ideia e critica a posição contrária.',
    ],
  },
  {
    label: 'Fórum',
    description: 'Discussão entre participantes em ambientes online, com linguagem informal e respostas a outros comentários.',
    prototypes: [
      'Alguém sabe como resolver isso? Tentei de tudo e nada funcionou, valeu pessoal!',
      'Concordo com o comentário acima, mas no meu caso aconteceu diferente, kkk.',
    ],
  },
  {
    label: 'Prosa/Lírico',
    description: 'Texto literário ou poético, com função estética: romances, contos, crônicas e poemas.',
    prototypes: [
      'Era uma vez uma menina que caminhava sozinha pela floresta ao cair da noite.',
      'Poema sobre saudade, amor e o mar, com versos, rimas e metáforas.',
    ],
  },
  {
    label: 'Jurídico',
    description: 'Documentos legais: leis, contratos, regulamentos, termos de uso e decisões judiciais.',
    prototypes: [
      'Art. 1º Fica estabelecido que as partes contratantes se obrigam ao cumprimento das cláusulas deste contrato.',
      'Lei, decreto ou regulamento com artigos, parágrafos e incisos sobre direitos e deveres.',
    ],
  },
  {
    label: 'Promocional',
    description: 'Texto publicitário que promove produtos, serviços ou marcas, com chamadas à ação.',
    prototypes: [
      'Aproveite! Compre agora com 50% de desconto e frete grátis, oferta por tempo limitado.',
      'Conheça o novo produto que vai transformar seu dia a dia. Garanta já o seu!',
    ],
  },
];
