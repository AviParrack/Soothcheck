# 2/12/24

- Microsoft shows how a suite-based approach allows them to be well positioned in multiple eras.

  - Which products in the suite are the most important changes, and new ones are added and the suite evolves in a new direction, but at every point the internal logic of the suite makes sense and new building blocks can be spun up for the new thing.

  - Microsoft is all about the suite being more important than any individual product.

  - This has allowed them to be well positioned in a number of successive eras; they can parlay that modular suite advantage and evolve what is in the suite and what the emphasis is more easily than they could evolve a single product.

  - Another example of the more resilience and adaptability that comes from a modular system than a monolithic one.

- What is the difference between a protocol and a platform?

  - Using a platform requires using code written by someone else.

    - It might be a tool (open source, perhaps running on your computer)

    - It might be a service (requiring reaching out to the code running on someone else's turf on an ongoing basis and for example presenting an API key).

    - But the platform code was almost certainly written by someone else.

  - A protocol doesn't *require* using code written by someone else.

    - A protocol is a structured communication procedure that has the intention of having multiple entities that can speak it.

    - Sometimes if there's a very prominent first mover, everyone else just makes themselves conform to that API, it becomes a kind of protocol, even though the initial mover didn't intend for it to be,

      - In that the *others* interpret it as a protocol and intend to have multiple speakers of it.

    - For protocols, you can always write your own code to speak your end of the protocol if you want to.

  - In things like [<u>Gardening Platforms</u>](https://komoroske.com/gardening-platforms) I use the word “platform” to describe what others might call “protocols”.

- It’s well established that humans have a strong anchoring bias. Why?

  - In a vast sea of possibilities, many of which are non-viable, we prefer ones that are more likely to be viable.

  - If an option is pre-existing then all else equal, it’s more likely to be viable.

    - Ideas that aren’t useful tend to erode over time when they aren’t used.

    - Entropy erodes things.

    - So if an idea is pre-existing, it either popped onto the scene recently, or someone found it useful enough to keep around.

  - The more times it's worked in the past, in the more varied the conditions, the more likely it is to work in this condition, too. It's [<u>Lindy</u>](https://en.wikipedia.org/wiki/Lindy_effect).

  - Streaks and precedents are repeated anchors over time, which become more and more of a rut.

  - When in doubt, start out where you were last time that worked (or didn't *not* work) and then consider if any adjacent moves feel better. If not, stay where you are.

  - The anchoring bias exists because as a general purpose decision making tool it’s pretty good in most circumstances, actually!

    - We just tend to focus on the edge cases where it doesn’t work very well, like adversarial negotiations.

- In systems that have a quality component (e.g. search engines, or LLMs), the query stream coevolves with the underlying quality of the service.

  - Users as a population clue into what it can do and give it queries it will do well at.

  - There's always some bizarro random queries that don't work, but the creators of the service will typically study those carefully and use that to decide which quality improvements to prioritize (a variant of "pave the cowpaths").

  - Over time, those random queries help extend the service in ways that other users come to expect to work, too.

  - When there's a huge first mover like ChatGPT, everyone else will be evaluated on how well they perform on queries optimized for ChatGPT primarily.

  - If services like Gemini don't do well on those, then users might say "eww not as good as ChatGPT" and not come back.

  - So services like Gemini have to be at least good enough on ChatGPT-style queries, and then actively great at some other differentiated things that some users might think to try.

- Boulders are obvious obstacles from a distance. Swarms are not, they just look like a diffuse cloud.

  - Imagine a conversation between a leader diagnosing slow execution and a line employee responsible for execution.

    - "Where's the big boulder that's making this hard?"

    - "There isn't one. There's a swarm of small issues, each of which requires independent coordination costs to chase down, convincing tons of different teams to do a thing that isn't a priority for them."

    - “Hmm, have you tried simply being more bold?”

  - You can’t heroically slay a coordination headwind, because it’s not one entity.

    - It’s more like quicksand. The faster you try to move, the more you get bogged down in it.

  - The coordination cost is extremely hard to see at a distance, so it’s often undercounted and hidden. But it’s extremely easy to *feel* up close.

- The tyranny of the rocket equation applies to organizations, too.

  - The [<u>Tsiolkovsky rocket equation</u>](https://en.wikipedia.org/wiki/Tsiolkovsky_rocket_equation) describes how much fuel is required to get a given mass of payload into orbit.

  - The “tyranny” part is that as you increase the desired payload, you need to start with more fuel, which increases the effective payload, which requires more fuel…

    - This is a non-linear dynamic leading to quickly escalating costs for even modest payload size increases.

  - I think the coordination headwind has a similar dynamic for organizations.

  - When you have a successful organization executing on a market opportunity, the obvious thing to do is add incremental heads to extract more, faster.

  - And incremental heads do help execute faster… but at a strongly sub-linear growth rate.

  - That’s because as you add more people, there are now more people that need to coordinate to get any plan decided, a cost that grows super-linearly.

  - As companies grow to mega size, the experience for employees gets increasingly kafka-esque, with everyone running in circles unable to get much done.

  - Big companies are a terrible way to get big things done in problem domains that have even tiny bits of uncertainty, and yet they are the only plausible way to increase output, so companies grow as big as their market opportunity demands.

  - This the tyranny of the coordination headwind.

  - I wonder if this is another thing that AI technologies might affect. If you can give a small number of employees significantly more leverage, you increase output without increasing coordination costs.

- High agency and autonomous execution are different.

  - They’re the same *kind* of thing, but at different scales.

  - High agency is about being able to pick from a broad range of possibilities.

  - Autonomous execution is about picking from a much smaller cone of possibilities.

  - A company like Apple has highly autonomous execution but does not encourage high agency for all but a very small number of people.

  - The more it’s about execution, the more it’s about metrics: line cooks.

  - The more it’s about agency, the more it’s about taste: chefs.

  - Line cooks are typically unable to effectively manage chefs.

    - Taste isn’t possible to measure directly with numbers.

    - The line cook will reflexively reach for things that can be measured, which is in tension with the things that create quality.

  - Chefs can *sometimes* effectively manage line cooks.

- Something that is "like clockwork" cannot be a creative act.

  - A creative act requires applying your agency without coercion.

  - If it's like clockwork, the agent is delegating their agency to the machine, to the clock. They are just downstream of that, an automaton.

  - You can cap downside with this approach, but you cannot create unexpected upside.

- When you optimize for high quality, the tradeoff is not cost, but scale.

  - A very high quality standard might not be *expensive* to produce per se, but it does set a ceiling on how broad/diverse/scaled an opportunity a team can plausibly cover.

  - And growing the team doesn’t help, because then it gets harder and harder to hold a coherent taste bar.

- Different pace layers have different rates of healthy process.

  - The higher the variance, the less process you want per unit.

  - The higher the pace layer, the higher the variance, so the less process you want.

  - A spectrum of contexts from low variance to higher variance.

    - A toaster factory (same model for a decade)

    - A car factory (new tweaked models each year)

    - Computer hardware

    - Fintech

    - Generic software

- Chris Dixon has a new book out about Crypto called *Read, Write, Own*.

  - I hear that one of the frames from the book is crypto-as-casino vs crypto-as-computer.

  - He asserts, as I think many people would, that they care about crypto-as-computer but don’t care at all for crypto-as-casino.

  - I wonder if they’re actually separable, though.

  - Crypto, in some ways, is the hardest possible environment to do platform design.

    - Maximal decentralization means maximal coordination costs.

    - The fact everything is directly financialized means any zero-day could drain billions of dollars in the blink of an eye.

    - It all adds up to massive headwinds that have to be overcome.

  - The casino part is what makes it worth the while for the vast majority of entities that invest time and effort in crypto.

  - So saying ‘I want crypto without the casino’ is hard for me to imagine, because the casino is the only reason crypto works at all!

  - This is not a dynamic unique to crypto, by the way.

  - I think you could reasonably frame all of Silicon Valley as a “computer” (sifting through a large number of bad ideas to find the small number of great ones) and a casino (massive paydays for the early investors in the ideas that turn out to be great).

  - One of the only ways to quickly find game-changing ideas is to have a system with a lot of variance (which implies, inevitably, a lot of crappy ideas), combined with over-the-top lottery style dynamics for finding the winners early.

- I think the world needs more compliments.

  - If you find yourself complimenting someone where they can’t see (e.g. noting a positive quality of theirs to a coworker), why not compliment them directly, too?

- Don’t take the credit if you wouldn’t want to take the blame.

- If your OODA loop is sufficiently faster than other players, you don't need to anticipate, you can just react.

- Steamrollers are dangerous when you’re directly in their path and unable to move out of the way, but they are easy to maneuver around.

  - The aggregator playbook is an extremely powerful steamroller… but is also easy to see its internal logic miles away, and sometimes there are asymmetric plays you know the aggregator will never do.

- Let go of the plan to make it more likely you end up in a place you love.

  - A plan is a single possibility; if any of the steps don’t work, the plan doesn’t work.

  - In practice there are a ton of unknowns and changing conditions.

    - Even if your plan would have worked when you started, by the time you’re mid-way through the context might have changed so it no longer works.

  - When a plan starts to not work, the default thing is to double down on it, to try it *harder*.

    - Plans can thus become liabilities if they turn out to be wrong.

    - If your plan ends up being a dead end, then you’re left in a place you don’t like.

  - If you instead have a general goal, you can orient yourself and make good decisions even in changing conditions.

  - You’ll continuously tend towards that goal, perhaps using a different path than you originally thought, based on changing conditions.

  - You’ll be less likely to end up precisely where you expected to, but you’ll be more likely to end up in a place you’re happy about.

- In complex contexts, don’t build plans, grow systems.

- Running an organization like a VC fund is a bad idea.

  - The VC logic is clear: invest in a diversity of bets, and then push each to have blockbuster-or-bust results.

    - As long as one turns out to be a blockbuster, it doesn’t matter if the rest went bust.

  - But a hidden assumption is that each bet is independent.

  - This logic is a terrible idea when you have lots of interdependent bets.

  - When you’re inside of a given org, the various products support and improve each other; a suite emerges that is greater than the sum of its parts.

    - This happens even if you didn’t intend it to or optimize for it.

    - And if it doesn’t, why are you building them under the same roof? Presumably there’s at least some reason they are better or cheaper together.

  - Under those circumstances, pushing each individual product to be a blockbuster-or-bust will end up killing off many of the things that made the whole significantly stronger.

- Specialists are more likely to go extinct than generalists.

  - An [<u>observation from evolutionary biology</u>](https://www.youtube.com/watch?v=HE2MmAGUgnw), but relevant in any ecosystem.

  - Similar logic to an assertion I made a few weeks ago about monoliths being more likely to go extinct than a system made of building blocks.

- Thinking in gray is hard.

  - We start off thinking in black and white, and it’s very comforting.

  - It’s much easier to be decisive in a black-and-white thinking mode.

  - All you need to do is identify which side is right (which, what a crazy random happenstance, tends to be whichever you started on), and you’re done.

  - It’s easy to think that people who constantly flip-flop clearly just aren’t as decisive as you.

  - But at a certain point the world will beat black and white thinking out of you.

  - The real world does not fit into clean black and white categorizations.

  - The real world is cruel and complex and unyielding.

  - At a certain point people tend to lose black and white thinking, and when they do it feels like a terrifying and disorienting loss.

- In a messy and untrustworthy world, there’s more friction for users to try a new thing.

  - Aggregators solved this problem by creating worlds unto themselves where they set the rules.

  - They can construct the rules so that things are safe and clean, and thus can be relatively low friction.

  - But this has a downside: they did it by handing god-like power to themselves.

    - An aggregator: “It’s my world, you’re just living in it.”

  - You might be able to solve the problem in a different way, with different laws of physics.

  - If you did, you could hand the power the aggregators usurped and hand it back to users.

- The entity that stores the state has the most power in a system.

  - I’ve asserted in the past that computing experiences are a combination of code + data. If you change either one, the identity of the experience changes.

  - I used to see app developers get tripped up by this a lot when thinking about in-app browsers on mobile devices.

    - “I want to add experiences to my browsing experience in my app, so I’ll use a WebView.”

    - “Yes, but then your users don’t get access to their primary cookie jar, and that’s the thing that makes the web experience useful to a user.”

    - In browsers, the cookie jar (well, technically the profile) is what stores login credentials, session cookies, as well as payments information.

    - Without the cookie jar, it’s a discontinuous experience for users, an island.

  - The entity that renders the final pixels on screen has a lot of power; but the entity that controls access to the stored data has even more.

- What's the superpower of the web?

  - The web is a fabric of computing that is on nearly every device beefy enough to run it.

  - It is open, so it works mostly the same everywhere it shows up.

  - And no one entity has unilateral power to define what the web can do.

    - Unless there were a computing device used by more than a billion users where the manufacturer disallowed other browser rendering engines, but that would be crazy, society would never stand for that.

  - But the real super power of the web is **links**.

    - Links teleport you, safely, to a totally new experience in an instant, carting along your cookie jar of state.

    - One of the key invariants in the security model is “tapping a link should never be dangerous”

  - This means that instead of each site being an island that has to be viable on its own, experiences are tied into a massive web of interrelated experiences.

  - The whole is significantly larger than the sum of its parts.

  - Apps are, in contrast, a collection of tiny monoliths that can’t be composed or easily navigated between.

  - Each app needs to stand on its own as an individually viable experience.

  - If AI is the internet, the web hasn’t been invented yet.

  - What will the AI-native web be?

- When a new tech paradigm bursts on the scene, it starts off messy and inventive.

  - An era of community gardening and tinkering.

  - Over time the swarm figures out patterns that work especially well, and those become more common and more financialized.

    - The reason to create something goes from “it’s cool” to “it makes money”

  - An era of factory farming, efficiency, and consolidation.

  - In the late stage, aggregators figure out how to use those efficient patterns to create a world unto themselves.

  - I don’t think of the web/app era as two eras, I think of it as one.

  - 2008 was the turning point where it pivoted from the growing / exploratory phase, to the consolidation / extraction phase.

  - 2008 was when the iPhone aggregator got going, and when Facebook’s aggregator had become a gravity well.

- Why do people contribute to Wikipedia?

  - Because everyone knows that everyone knows that Wikipedia is the canonical source of factual information on the web.

  - This is an effect that has gravity-well dynamics; it accelerates as it goes.

  - If you could rewind the clock, it’s not a given that wikipedia.org would be the undisputed winner, but it is a given, in my opinion, that something like wikipedia would happen.

  - I wrote my undergraduate thesis on the emergent power dynamics of Wikipedia’s user community.

    - One editor I interviewed was retired, and every day he’d go to the local library and transcribe a few articles on fish species from an out-of-copyright book.

    - The information was at the time nowhere else on the internet.

    - It was a point of pride to him that he was the one bringing that information online.

    - And where else would he post that information than on Wikipedia?

  - A similar dynamic: a few years ago I worked adjacent to User Street View on Google Maps.

    - Users could upload panoramas they took, and in limited circumstances they’d show up in Google Maps.

    - There was one example where a particularly motivated user bought a few GoPros and duct taped them to a broomstick and then drove around his small island nation to collect panoramas.

      - His nation was one that hadn’t been prioritized by our official data collection process.

    - Someone in leadership commented “He must really love Google Maps!”

    - To which I replied: “No, he really loves his *nation*. He wants to put it on the map, and the map that everyone uses is Google Maps, so that’s where he puts it.”

  - Editors on Wikipedia are a very small number of highly engaged people who have devoted significant time and effort to Wikipedia, which confers them authority.

    - If the only way to climb the hill is to do the expensive/valuable thing, then being at the top of the hill does imply a legitimate authority.

  - The Wikipedia article's talk page is a great record of the most interesting / relevant / incisive discussions on the internet on that topic.

    - Nowhere else can you see, so perfectly distilled in one place, the constantly-evolving knife’s edge of society’s collective understanding of a given topic.

- I saw an excellent talk from [<u>Thalis Wheatley</u>](https://pbs.dartmouth.edu/people/thalia-wheatley) at SFI last week.

  - Here are my notes, so condensed as to be practically a caricature of her findings!

  - They did brain scans of new undergraduates the first week of school, and based on that were able to predict, with some small but significant degree of accuracy, which people would be friends months later.

  - They did another experiment where they had people watch confusing clips of a movie without sound while in an MRI.

    - Then all of the participants got to talk about it with others in the experiment, and then went back in to watch the clips again.

    - The MRIs found previously-random neural responses had largely synchronized after the conversations.

    - That synchronization extended even to novel clips from the movie that they hadn't discussed.

  - They also used LLMs/embeddings to map the "turns" that a conversation takes on topics to get a fingerprint of how they evolve over the life of the discussion.

  - She talked about how their framework they had synthesized about good / productive conversations was 1) co-created, 2) collectively steered, 3) agenda free, and 4) open-ended.

  - She also said that in their research strangers in conversation tend to start with topics like the weather because a) it's not socially charged (unlikely to make the other person angry) and b) they have a high degree of in/out degre (lots of adjacent topics to move into from it easily)

- If you want to maximize upside, hire creative people and give them space.

  - If you want to minimize downside, hire operators and put them in a stable structure to execute.

- General purpose career advice from a long-term editor at New Yorker:

  - “It’s harder than you think.

  - Both things are true.

  - I’ll tell you when you’re older.”

- When something is intuitive, you don’t have to “think about it” for it to work.

  - At the Exploratorium, there’s an exhibit where you have to time the firing of four leg muscles to make them turn a bicycle wheel.

  - It’s nearly impossible, but using our legs to turn a real bicycle wheel is something we could do any day without thinking.

  - Watching my 2 year old son try to learn how to pedal his tricycle underlines how non-intuitive it is to start.

  - But once you successfully get it going, it feels like the most natural thing in the world, something in your muscle memory, in your bones, something you don’t need to spend any cycles thinking about: it’s intuitive.

  - A thing that has become intuitive for you is something that is hard to teach to others, because you can’t consciously access the knowhow to explain it, you only know the feel of it.

- Sometimes the most important things to see are like a magic eye picture.

  - It’s right in front of you, but impossible to see without defocusing your eyes just right.

  - Once you’ve practiced and it’s intuitive to you, it will feel like the most natural thing in the world. “Look, the shark is *right there!”*

  - But other people without the practice will say “All I see is a bunch of dots. Are you sure you aren’t making it up?”

  - Magic eye pictures require practice, the time to take a step back and defocus.

    - This feels like exactly the wrong thing to do; the busier people are, the less likely they are to see the picture if they can’t already see it.

  - This is one of the reasons it’s so hard to get people to see things that are in their blind spot.

- Our brains are very good at melding with tools, incorporating them into our mental map as an intuitive extension of ourselves and our agency.

  - The shorter the feedback loop / lower the error between intention and outcome, the faster and higher fidelity the merge.

- I bought a Vision Pro. The dinosaur encounter makes everyone who tries it giggle like a school kid.

  - It’s an example of a condition where your rational brain knows it’s fine, but your reptile brain thinks “holy crap this thing is going to eat me!”

  - The dissonance between the two subsystems creates a thrill.

  - Same logic applies for haunted houses, horror movies, and roller coasters.

- As the last tech era consolidated into a small number of powerful aggregators, users have learned that many valuable services should be free.

  - This makes it very hard for new entrants to break in; they need to have a business where they make a lot of money elsewhere to subsidize the consumer aggregation.

  - LLMs cost money to do inference on; the entity that pays for those costs will have a lot of staying power.

  - The user will have to trust the incentives of whoever pays for their LLM completion costs.

  - If the paying entity primarily makes money from advertising, it will be hard for users to ever fully trust any advice the system gives them.

- We tend to look at products in isolation, but they are coevolved with their surrounding context.

  - If you're trying to build a product at a higher pace layer, but it has to fit into a lower, slower pace layer, you'll get bogged down.

  - You need to go as slow as the slowest pace layer you interact with that could kill you if you do it wrong.

  - In general, if you implicitly think you're at a different pace layer than you are, you're going to have a bad time.

    - If you think you're at a lower one than you are, something in the environment will go faster than you and kill you before you know what's happening.

    - If you think you're at a higher one than you are, you'll be trying to sprint through molasses, perhaps making non-viable decisions.

- r-selected and k-selected is a matter of perspective.

  - In biology, r-selected organisms are ones that have a large number of cheap offspring, hoping that a small number will survive.

  - K-selected organisms have a small number of expensive offspring and invest heavily in them.

  - VC funding is r-selected from the point of view of the VC (any one of dozens can survive and it's fine), but k-selected from the point of view of the startup’s founder (an n count of 1 that they are deeply tied to).

- Slightly unrealistic assumptions can compound into an absurdly unrealistic outcome.

  - This is true if they multiply in a series.

  - In this situation, the right argument is not "each step sounds reasonable so therefore the conclusion is too" it's "the end point doesn't pass the laugh test so it doesn't matter if the inputs are all slightly reasonable".

- A system can have closed components and still be open, and many open components and still be closed.

  - The question is, can the user and the ecosystem route around any one component if they want to? Do users have meaningful agency on how to combine the components?

  - If the way the user interacts with the whole system is mediated by one single provider at the top (often an aggregator) then they can't escape the value judgements and decisions of the provider; the system is closed, even if many components of it are "open".

  - But imagine a system where the user is empowered to mix and match and stitch together many components in any way they see fit.

  - Even if many of the components they're stitching together are individually closed, the system is still meaningfully open.

  - The openness of a system is primarily determined by the configurability and combinability of the layer closest to the user.

- Never cut off your wings.

  - Your wings are what allow you to leave the ground and fly, to do great things.

  - Everyone has slightly different wings; their particular super powers.

  - Once you cut off your wings, you can no longer do those great things.

  - The machine you are in will take an instrumentalist view to you because it must: "I need this box filled, this person is sitting next to this box, they don't fit because of those things on their back, so to fill the box I need them to cut off their wings."

  - Someone who takes a non-instrumentalist approach to you will never ask you to cut off your wings.

- We maintain mental models of the systems that we interact with that can be wrong, and sometimes confuse us.

  - Matt Webb told me he had heard somewhere that humans tend to categorize objects into four basic categories:

    - Rocks - No movement or agency; just obey the laws of physics.

    - Plants - Alive, but fully reactive / slow; you can garden them.

    - Animals - Alive and active, with some degree of agency but strictly dumber than us. We can trick them without having to be particularly clever.

    - People - Alive and with a level of intelligence and agency roughly equivalent to ours; we need to have a full-fledged theory of mind to deal with them.

  - Waymo cars feel like animals.

  - Raw AI foundation models feel something like plants.

  - ChatGPT feels like a person.

- A few stray thoughts on LLMs.

  - I love using ChatGPT as a kind of family feud "what will the average (X category of person) think about this phrase".

    - Kind of an automatic wisdom of the crowds.

  - If you have one AI, it's got to be boring as hell.

    - If you have lots of AI agents, they can have very different personalities.

  - LLMs are a theory-of-mind in a box, magical duct tape you can put onto anything to help it interact with things in the people category.

    - We tend to assume a human-style intelligence in systems we categorize as a person. Which can cause us to mis-categorize computers.

    - Someone told the story from 30 years ago of a kid using an 8 bit computer and typing into a file that the OS expected to be a BASIC program the text "What is the capital of Idaho." When it gave an error, the kid said, "See? it's dumb."

    - LLMs are absurdly good at behaving like they have a coherent human mind, so we figure, implicitly, that they are like humans in a lot of ways, but actually they're way better than humans in some areas (rough vibes based fact recall), and way worse than humans in others (multi-ply theory of mind and rational thought).