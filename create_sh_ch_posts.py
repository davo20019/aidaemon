import json
import os
from datetime import date

base_path = "/Users/davidloor/projects/learnenglishsounds/app/blog/posts/"
today = date.today().isoformat()

# Content for Spanish
es_slug = "diferencia-pronunciacion-sh-ch-ingles"
es_content = """
<p>Una de las confusiones más comunes para los hispanohablantes al aprender inglés es la diferencia entre los sonidos <strong>SH /ʃ/</strong> y <strong>CH /tʃ/</strong>. En español, tenemos el sonido "CH" (como en "chico"), pero el sonido "SH" solo existe en ciertas variantes regionales. Esto hace que muchos estudiantes pronuncien "shoes" como "choose" o "wash" como "watch", cambiando completamente el significado de lo que quieren decir.</p>

<p>¡No te preocupes! La diferencia física es sencilla una vez que la conoces.</p>

<h2>El sonido SH /ʃ/: Silencio continuo</h2>
<p>El sonido <strong>/ʃ/</strong> es el mismo que hacemos cuando pedimos silencio: "¡Shhh!".</p>
<p><strong>Cómo hacerlo:</strong></p>
<ul>
<li>Junta los dientes pero sin apretarlos.</li>
<li>Redondea los labios ligeramente hacia afuera.</li>
<li>Levanta la parte media de la lengua hacia el paladar, pero sin tocarlo.</li>
<li>Deja salir el aire de forma <strong>continua</strong> y suave. No hay interrupción.</li>
</ul>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="she" data-ipa="ʃiː" data-definition="ella" data-example-sentence="She is my friend." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="shop" data-ipa="ʃɒp" data-definition="tienda" data-example-sentence="I need to go to the shop." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<h2>El sonido CH /tʃ/: Explosivo</h2>
<p>El sonido <strong>/tʃ/</strong> es muy parecido al español "ch" en "chocolate". A diferencia del SH, este sonido es <strong>explosivo</strong>.</p>
<p><strong>Cómo hacerlo:</strong></p>
<ul>
<li>Empieza con la lengua tocando el paladar justo detrás de los dientes superiores (como si fueras a decir una 't').</li>
<li><strong>Bloquea</strong> el aire completamente por un instante.</li>
<li>Suelta el aire de golpe. Es como un estornudo: "¡A-chú!".</li>
</ul>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="cheese" data-ipa="tʃiːz" data-definition="queso" data-example-sentence="I love cheese pizza." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="chop" data-ipa="tʃɒp" data-definition="picar/cortar" data-example-sentence="Chop the onions finely." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<h2>Pares Mínimos: SH vs. CH</h2>
<p>La mejor forma de practicar es con "pares mínimos": palabras que solo se diferencian por este sonido.</p>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="sheep" data-ipa="ʃiːp" data-definition="oveja" data-example-sentence="The sheep is white." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="cheap" data-ipa="tʃiːp" data-definition="barato" data-example-sentence="This shirt was very cheap." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="wash" data-ipa="wɒʃ" data-definition="lavar" data-example-sentence="Wash your hands." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="watch" data-ipa="wɒtʃ" data-definition="mirar/reloj" data-example-sentence="Watch the movie." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="shoes" data-ipa="ʃuːz" data-definition="zapatos" data-example-sentence="My shoes are new." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="choose" data-ipa="tʃuːz" data-definition="elegir" data-example-sentence="Choose one color." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<h2>Resumen</h2>
<p>Recuerda: <strong>SH</strong> es suave y continuo (fluye). <strong>CH</strong> es duro y explosivo (se detiene y explota).</p>
"""

es_data = {
    "slug": es_slug,
    "title": "Diferencia entre SH y CH en inglés: Guía de Pronunciación",
    "date": today,
    "excerpt": "¿Confundes 'shop' con 'chop'? Aprende la diferencia clave entre los sonidos /ʃ/ y /tʃ/ en inglés con ejemplos prácticos para hispanohablantes.",
    "content": es_content,
    "category": "Pronunciation",
    "tags": ["pronunciation", "consonants", "sh sound", "ch sound", "common mistakes", "spanish speakers"],
    "lang": "es",
    "keywords": "pronunciación sh ch inglés, diferencia sh ch, pronunciar sh ingles, pronunciar ch ingles, errores comunes pronunciación",
    "author": "David Loor",
    "dateModified": today,
    "status": "published"
}

# Content for Portuguese
pt_slug = "diferenca-pronuncia-sh-ch-ingles"
pt_content = """
<p>Uma confusão comum para falantes de português ao aprender inglês é a diferença entre os sons <strong>SH /ʃ/</strong> e <strong>CH /tʃ/</strong>. Embora tenhamos o som de "ch" (chiado) em muitas regiões do Brasil (como em "tia" em alguns sotaques ou "chave"), o som explosivo do inglês /tʃ/ pode ser confundido com o /ʃ/ contínuo, ou vice-versa, dependendo da palavra.</p>

<p>Saber a diferença é crucial para não pedir "shoes" (sapatos) quando você quer "choose" (escolher)!</p>

<h2>O som SH /ʃ/: O Silêncio</h2>
<p>O som <strong>/ʃ/</strong> é aquele que fazemos para pedir silêncio: "Shhh!". É um som contínuo e suave.</p>
<p><strong>Como fazer:</strong></p>
<ul>
<li>Junte os dentes levemente.</li>
<li>Arredonde os lábios para a frente.</li>
<li>Deixe o ar sair de forma <strong>contínua</strong>. Você pode segurar esse som o tempo que tiver fôlego: "Shhhhhhh".</li>
</ul>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="ship" data-ipa="ʃɪp" data-definition="navio" data-example-sentence="The ship is huge." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="cash" data-ipa="kæʃ" data-definition="dinheiro (espécie)" data-example-sentence="I only have cash." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<h2>O som CH /tʃ/: O Espirro</h2>
<p>O som <strong>/tʃ/</strong> é explosivo. É parecido com o som de "tchau" ou um espirro "Atchim!". Ele começa bloqueando o ar.</p>
<p><strong>Como fazer:</strong></p>
<ul>
<li>Comece com a língua tocando o céu da boca atrás dos dentes (posição do T).</li>
<li><strong>Bloqueie</strong> o ar totalmente por uma fração de segundo.</li>
<li>Solte o ar com força em um chiado. O som é curto e não pode ser prolongado.</li>
</ul>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="chip" data-ipa="tʃɪp" data-definition="batata frita/chip" data-example-sentence="Fish and chips." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="catch" data-ipa="kætʃ" data-definition="pegar" data-example-sentence="Catch the ball!" data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<h2>Pares Mínimos: SH vs. CH</h2>
<p>Pratique a diferença com estas palavras parecidas:</p>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="sheep" data-ipa="ʃiːp" data-definition="ovelha" data-example-sentence="Look at the sheep." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="cheap" data-ipa="tʃiːp" data-definition="barato" data-example-sentence="It was very cheap." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="share" data-ipa="ʃɛr" data-definition="compartilhar" data-example-sentence="Share your toys." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="chair" data-ipa="tʃɛr" data-definition="cadeira" data-example-sentence="Sit on the chair." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
<word-practice-card data-word="wish" data-ipa="wɪʃ" data-definition="desejar" data-example-sentence="Make a wish." data-example-ipa="" data-lang="en"></word-practice-card>
<word-practice-card data-word="witch" data-ipa="wɪtʃ" data-definition="bruxa" data-example-sentence="The witch flies on a broom." data-example-ipa="" data-lang="en"></word-practice-card>
</div>

<h2>Resumo</h2>
<p>A regra de ouro: <strong>SH</strong> flui (suave). <strong>CH</strong> explode (forte). Se você consegue segurar o som ("Shhhhh"), é SH. Se não consegue, é CH.</p>
"""

pt_data = {
    "slug": pt_slug,
    "title": "Diferença entre SH e CH em Inglês: Guia de Pronúncia",
    "date": today,
    "excerpt": "Você confunde 'sheep' com 'cheap'? Entenda a diferença entre os sons /ʃ/ e /tʃ/ em inglês e melhore sua pronúncia com exemplos práticos para falantes de português.",
    "content": pt_content,
    "category": "Pronunciation",
    "tags": ["pronunciation", "consonants", "sh sound", "ch sound", "common mistakes", "portuguese speakers"],
    "lang": "pt",
    "keywords": "pronúncia sh ch inglês, diferença sh ch, pronunciar sh, pronunciar ch, erros comuns pronúncia inglês",
    "author": "David Loor",
    "dateModified": today,
    "status": "published"
}

# Write files
with open(os.path.join(base_path, es_slug, "index.json"), "w", encoding="utf-8") as f:
    json.dump(es_data, f, ensure_ascii=False, indent=2)

with open(os.path.join(base_path, pt_slug, "index.json"), "w", encoding="utf-8") as f:
    json.dump(pt_data, f, ensure_ascii=False, indent=2)

print("Files created successfully.")
