import re


def replace_digits(str):
    return re.sub('[0-9]', ' ', str)


temp = replace_digits(
    "a1about2above3across4after5afterwards6again7against8all9almost10alone11along12already13also14although15always16am17among18amongst19amoungst20an21and22another23any24anyhow25anyone26anything27anyway28anywhere29are30around31as32at33be34became35because36been37before38beforehand39behind40being41below42beside43besides44between45beyond46both47but48by49can50cannot51could52dare53despite54did55do56does57done58down59during60each61eg62either63else64elsewhere65enough66etc67even68ever69every70everyone71everything72everywhere73except74few75first76for77former78formerly79from80further81furthermore82had83has84have85he86hence87her88here89hereabouts90hereafter91hereby92herein93hereinafter94heretofore95hereunder96hereupon97herewith98hers99herself100him101himself102his103how104however105i106ie107if108in109indeed110inside111instead112into113is114it115its116itself117last118latter119latterly120least121less122lot123lots124many125may126me127meanwhile128might129mine130more131moreover132most133mostly134much135must136my137myself138namely139near140need141neither142never143nevertheless144next145no146nobody147none148noone149nor150not151nothing152now153nowhere154of155off156often157oftentimes158on159once160one161only162onto163or164other165others166otherwise167ought168our169ours170ourselves171out172outside173over174per175perhaps176rather177re178same179second180several181shall182she183should184since185so186some187somehow188someone189something190sometime191sometimes192somewhat193somewhere194still195such196than197that198the199their200theirs201them202themselves203then204thence205there206thereabouts207thereafter208thereby209therefore210therein211thereof212thereon213thereupon214these215they216third217this218those219though220through221throughout222thru223thus224to225together226too227top228toward229towards230under231until232up233upon234us235used236very237via238was239we240well241were242what243whatever244when245whence246whenever247where248whereafter249whereas250whereby251wherein252whereupon253wherever254whether255which256while257whither258who259whoever260whole261whom262whose263why264whyever265will266with267within268without269would270yes271yet272you273your274yours275yourself276yourselves277")

function_words = {"the", "which", "still", "although", "forty", "and", "up", "last", "past", "nobody", "of", "out",
                  "being", "himself", "unless", "to", "would", "must", "seven", "mine", "a", "when", "another", "eight",
                  "anybody", "I", "i", "\"", ")", "(", "'s", ".", ",", "for", "of", "your", "between", "along", "till",
                  "in", "will", "might", "round", "herself", "you", "an", "''", "``", ";", "-", "]", "[", "him",
                  "their", "both", "several", "twelve", "that", "who", "five", "someone", "fifteen", "it", "some",
                  "four", "whatever", "beyond", "for", "two", "around", "among", "whom", "he", "because", "while",
                  "across", "below", "on", "how", "each", "behind", "none", "we", "other", "under", "million", "nor",
                  "they", "could", "away", "outside", "more", "be", "our", "every", "nine", "most", "with", "into",
                  "next", "thousand", "this", "these", "anything", "shall", "have", "than", "few", "myself", "but",
                  "any", "though", "themselves", "as", "where", "since", "itself", "not", "over", "against", "somebody",
                  "at", "back", "second", "upon", "what", "first", "nothing", "thirty", "so", "much", "without",
                  "third", "there", "down", "during", "above", "or", "its", "six", "therefore", "one", "should",
                  "enough", "everybody", "by", "after", "once", "towards", "from", "those", "however", "thus", "all",
                  "may", "half", "everyone", "she", "something", "yet", "near", "no", "three", "whether", "inside",
                  "his", "little", "everything", "nineteen", "do", "many", "until", "yourself", "can", "why", "hundred",
                  "fifty", "if", "before", "within", "whose", "about", "such", "ten", "anyone", "my", "off", "twenty",
                  "per", "her", "through", "either", "except", "START", "END", 'a', 'about', 'above', 'across', 'after',
                  'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although',
                  'always', 'am', 'among', 'amongst', 'amoungst', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
                  'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'be', 'became', 'because', 'been',
                  'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both',
                  'but', 'by', 'can', 'cannot', 'could', 'dare', 'despite', 'did', 'do', 'does', 'done', 'down',
                  'during', 'each', 'eg', 'either', 'else', 'elsewhere', 'enough', 'etc', 'even', 'ever', 'every',
                  'everyone', 'everything', 'everywhere', 'except', 'few', 'first', 'for', 'former', 'formerly', 'from',
                  'further', 'furthermore', 'had', 'has', 'have', 'he', 'hence', 'her', 'here', 'hereabouts',
                  'hereafter', 'hereby', 'herein', 'hereinafter', 'heretofore', 'hereunder', 'hereupon', 'herewith',
                  'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'i', 'ie', 'if', 'in', 'indeed',
                  'inside', 'instead', 'into', 'is', 'it', 'its', 'itself', 'last', 'latter', 'latterly', 'least',
                  'less', 'lot', 'lots', 'many', 'may', 'me', 'meanwhile', 'might', 'mine', 'more', 'moreover', 'most',
                  'mostly', 'much', 'must', 'my', 'myself', 'namely', 'near', 'need', 'neither', 'never',
                  'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere',
                  'of', 'off', 'often', 'oftentimes', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others',
                  'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'per', 'perhaps',
                  'rather', 're', 'same', 'second', 'several', 'shall', 'she', 'should', 'since', 'so', 'some',
                  'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'still', 'such',
                  'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there',
                  'thereabouts', 'thereafter', 'thereby', 'therefore', 'therein', 'thereof', 'thereon', 'thereupon',
                  'these', 'they', 'third', 'this', 'those', 'though', 'through', 'throughout', 'thru', 'thus', 'to',
                  'together', 'too', 'top', 'toward', 'towards', 'under', 'until', 'up', 'upon', 'us', 'used', 'very',
                  'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                  'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                  'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'whyever', 'will', 'with', 'within',
                  'without', 'would', 'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'}


def replace_commas(str):
    return re.sub(',', ';', str)


print(replace_commas("('luxury', 'compmod', 'FROM-ME')"))

print("('luxury', 'compmod', 'FROM-ME')".replace(",", ";"))
