// Lista de stopwords do NLTK (português) + complementos comuns e um conjunto
// reduzido em inglês, já que textos técnicos costumam misturar os dois idiomas.
const PT_NLTK = `a à ao aos aquela aquelas aquele aqueles aquilo as às até com como da das de dela delas dele deles
depois do dos e é ela elas ele eles em entre era eram éramos essa essas esse esses esta está estamos estão estar estas
estava estavam estávamos este esteja estejam estejamos estes esteve estive estivemos estiver estivera estiveram
estivéramos estiverem estivermos estivesse estivessem estivéssemos estou eu foi fomos for fora foram fôramos forem
formos fosse fossem fôssemos fui há haja hajam hajamos hão havemos haver hei houve houvemos houver houvera houverá
houveram houvéramos houverão houverei houverem houveremos houveria houveriam houveríamos houvermos houvesse houvessem
houvéssemos isso isto já lhe lhes mais mas me mesmo meu meus minha minhas muito na não nas nem no nos nós nossa nossas
nosso nossos num numa o os ou para pela pelas pelo pelos por qual quando que quem são se seja sejam sejamos sem ser
será serão serei seremos seria seriam seríamos seu seus só somos sou sua suas também te tem tém temos tenha tenham
tenhamos tenho terá terão terei teremos teria teriam teríamos teu teus teve tinha tinham tínhamos tive tivemos tiver
tivera tiveram tivéramos tiverem tivermos tivesse tivessem tivéssemos tu tua tuas um uma você vocês vos`;

const PT_EXTRA = `pois ainda sobre assim onde cada outro outra outros outras todo toda todos todas pode podem podendo
sendo sido desta deste destas destes dessa desse dessas desses nesta neste nestas nestes nessa nesse nessas nesses
naquela naquele tal tais apenas porque porém então vez vezes bem aqui lá ali através partir além sob qualquer
quais cujo cuja cujos cujas uns umas etc ser ter seja tão quanto quanta quantos quantas contudo entanto todavia
desde durante após antes perante mediante conforme segundo enquanto embora caso sim pra pro dum duma
aí cá lo la los las daí disso disto daquilo nisso nisto naquilo àquele àquela àquilo dela dele`;

const EN = `the of and to in a is for on with as by an be are this that it its from or at was were been has have had
not but which can will their they them these those than then there such into also more most other some what when
where who how our we you your i he she his her do does did so if no may should would could about between each`;

// resíduos de URLs e referências bibliográficas
const WEB = `http https www com br org net html htm pdf doi`;

export const STOPWORDS: ReadonlySet<string> = new Set(
  `${PT_NLTK} ${PT_EXTRA} ${EN} ${WEB}`.split(/\s+/).filter(Boolean),
);
