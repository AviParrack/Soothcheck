# 4/22/24

- Every new tech paradigm, in the Carlotta Perez sense, has a new radically lower-cost-than-before input.

  - The low-cost input for this next paradigm is LLMs.

  - LLMs look expensive compared to normal compute.

  - But they look *radically* cheap compared to generic human mental labor.

- Machine intuition is super useful as magical duct tape!

  - These are machine intuition but we use them for machine *reasoning*.

  - LLMs seem like they’re reasoning, but it’s really just absurdly good intuition and pattern matching.

  - It's a miracle that works at all as reasoning.

    - That’s an indication of how insanely good they are at extracting implied patterns.

    - Then again, maybe that says more about us than we'd like to admit.

    - Maybe most of *our* reasoning is the LLM style good-enough vibes matching.

  - LLMS are great at vibing their way to something approximately correct for anything with a vaguely grammatical structure (of any complexity).

- Everyone doing cool use cases with AI is realizing they’re impossible to package and distribute as stand-alone apps.

  - They're too inflexible when they don’t work.

  - Too expensive to package up a kernel of a use case into an app.

  - Too vanilla without deep context and data, (which a new standalone app starts from by default).

- What is the “same origin model”?

  - I talk about it here just about every week, but if you haven’t ever worked in building a browser you might not know what it means.

  - The same origin model is the fundamental laws of physics behind the web.

  - It has the following basic characteristics:

    - 1\) Data is segregated by origin.

      - An “origin” in the web is essentially a domain.

      - In the app model it’s an app.

    - 2\) Each origin starts with no data.

      - This is what makes visiting a new origin safe: it has no information on you.

      - A user can *choose* to import data into an origin.

        - This might be “the actions you take within the origin’s view” (e.g. text you type in it, or things you click on)

        - Or it might be things like “files you upload”

    - 3\) Data may flow freely within an origin.

      - There are no internal “border crossings” or permission prompts.

    - 4\) An origin can *choose* to export data out to another origin.

      - But by default origins can’t see any other origin’s data.

      - That is, origins have distinct, strong boundaries between them.

  - The same origin model is a simple, clear model that is easy to administer.

  - The problem is the downstream implications of this model.

- Apps are more similar to websites than to traditional desktop applications.

  - Apps, like websites, are segregated by origin and sandboxed.

    - Although apps have a richer set of high-fidelity APIs.

  - Traditional desktop applications, on the other hand, can read and write to the shared filesystem.

    - This allows them to coordinate with other applications.

    - But it also means you have to trust them more, because they could harm you more.

    - It’s no accident that most productive work today is still done on desktops, which by default don’t have the same origin cage.

  - It’s kind of funny that “app” and “application” sound similar but have wildly different emergent properties!

- The same origin model is what enabled the web and apps to exist.

  - Without it, they would not be possible.

  - But the same origin model is also the original sin.

  - The same origin model separates every origin into its own isolated pocket universe.

  - Origins that have a critical mass of end user engagement will tend to accumulate more data in than they emit out.

    - This leads to a significant preferential attachment effect, where the biggest apps tend to get even bigger.

    - As a pocket universe gets more mass, it attracts ever more mass.

  - The same origin model is what leads to accelerated centralization and an inexorable pull towards massive, all-encompassing, one-size-fits-all software.

- A sea of confusing permission prompts is downstream of the same origin model.

  - We take for granted is “just how it has to work”. But that’s not the case!

  - It’s actually a hack that is downstream of the decision to use a simple, high-contrast same origin model.

  - The same origin model requires a small set of extremely clear, discontinuous boundaries around the edges of the origin.

  - When data crosses the boundary (e.g. you give the origin the ability to turn on your camera, which allows data to flow inside), you need a permission dialog: a border crossing.

  - It’s possible to imagine a security model that allows safe composition more granularly, where the boundaries between things could get fractally more nuanced, and overall feel less like a high-contrast boundary and more like a gradient.

- You could use the same origin model to slingshot beyond it.

  - What if there was an origin that had its own open, gravity well dynamics?

  - Within that origin, you could have a more nuanced way of keeping track of valid compositions.

    - An alternate law of physics in the pocket universe within the larger universe it’s embedded in.

  - The pocket universe could grow into its own full universe that could come to have more power than the universe it’s embedded in.

  - Think of the galaxy in a marble that’s the macguffin in the original *Men In Black*.

- Privacy is about boundaries.

  - A society that was fully transparent could not work.

  - No variation would be possible

    - Everything would be pulled back towards the average.

    - “Why are you doing it differently than everyone else expects it to be done?”

  - You’d also have a background cacophony.

    - Making it impossible for anything to stand out from the noise and create motion or difference.

  - An overwhelming white noise that nothing could stand out from.

  - In complex adaptive systems, boundaries *must* emerge between components to prevent this heat death of the system and allow useful gradients of potential energy.

- I’m a fan of the “local first” ethos for software.

  - But I think it’s only a part of the problem.

  - The local first movement observes that our data is put in the cage of a 3rd party server.

    - By bringing it local first, we increase our agency over it, and make it harder for the data to be used in ways we don’t like.

  - But data isn’t just in a 3rd party server cage.

  - It’s also in the *origin’s* cage.

  - Data has combinatorial possibility.

    - When combined with other data, it can create new value.

  - The unconstrained combination outside of your turf is dangerous (e.g. advertisers).

    - When someone else does the combinations, they own the upside… which might be upside to them that’s downside to you.

  - But if you could combine your own data, within policies you control, you could unleash more potential energy from your own data, in a way that would benefit only you.

  - Let’s free our data from the origin that happened to create it!

- An app is an agglomeration of use cases surrounded by a defensible business model.

  - Apps are organized primarily by viable business models, not by amount of user value.

  - The same origin model creates a cave where the origin’s owner can hoard things.

    - For example, hoarding a bunch of user data they can rent out to advertisers.

  - What if we had a system with no place for the origin to hoard the data ?

- When a pixel perfect app doesn't work the way you want it to, you have no recourse except to stop using it.

  - A pixel perfect UI *demands* that you not change it

    - "You can interact with me on my terms or not at all"

  - For users, a passive stance.

  - But when a malleable thing doesn't work, you can tweak and fix it.

  - An active stance.

  - Does your tool encourage users to take an active or passive stance?

- Malleable systems can be made out of rigid building blocks.

  - If you have the agency to mix and match rigid building blocks, then you can assemble a malleable combination.

  - This is true even if you don’t have agency over the individual pixel-perfect / rigid building blocks.

  - In a given system, when you cannot compose the building blocks, your agency is reduced.

  - A key dimension: how big are the building blocks relative to the overall assemblage?

    - The larger the building blocks on a relative basis, the less agency you have as a user.

  - Apps are like big duplo blocks; hard to combine in any but the most rudimentary of ways.

  - Malleable software will have components that feel like sand: rigid building blocks, but so small that the overall thing flows like a fluid.

- When building an early stage product the single most important thing is to constantly be talking to customers.

  - An open ecosystem allows you to be constantly talking to users in the community by default.

  - Instead of needing to reach out to them, they come to you!

- A goal in any system: minimize nasty surprises.

  - For example, your data showing up in an unexpected place would be a nasty surprise.

  - A nasty surprise is something that violates the user’s implied mental model, and might cause them to never want to use the service again.

  - When you’re developing a new assistive service, you want to minimize how many people have a nasty surprise, while maximizing how many users/use-cases happen overall.

- Tools like Information Flow Control have existed for decades.

  - They allow you to make formal statements about the confidentiality and integrity of a composed system.

  - Part of the challenge is that they require precise policy definitions of when for example, certain information may be declassified (that is, safely used in a different context).

  - Writing a simple one-size-fits-all policy in a vacuum is easy, but then they are hard to use in real-world contexts.

  - If you want the policies to be applied to fractally wrinkled real-world situations, it could quickly get unwieldy.

    - Imagine a toy example of a cake recipe that in the third step calls for the addition of 5 tablespoons of tabasco sauce, and the policy needing to decide if that’s reasonable.

    - You’d need a policy like “spicy sauces are OK to be in recipes as long as the dish is a savory one and the overall amount of sauce is less than 3% of the total volume of the dish”.

    - It’s impossible to imagine such detailed policies to be created, especially when you imagine all of the real-world scenarios it needs to cover.

    - This reduces to the metacrap fallacy.

  - One way to think of this is to make this machine work you need to have created hyper-intricate, hyper-precision gears for every possible need.

    - Clearly impossible!

  - But there’s another way to do this.

  - There are lots of policies where 99% of the population would agree that it was allowed or not allowed.

    - For example, whether 5 tablespoons of tabasco sauce is legitimate in a cake recipe.

  - LLMs are society-scale crystallized intuition.

  - You can ask the LLM: “is it reasonable for a cake recipe to call for 5 tablespoons of tabasco sauce?”

  - That gives you an immediate good-enough default policy for cases where the vast majority of people would agree.

    - Good enough policies are ones that lead to very very few nasty surprises in practice, and where users that want to be a bit more flexible can ask the system to add a wrinkle for them.

    - There are a lot of plausible judgment calls where different people might disagree, but those are many, many orders of magnitude less common than the space of policies where the vast majority of people would agree.

  - If you have a good-enough baseline based on the crystallized intuition of society, you can wrinkle it with more specific needs.

    - For example, maybe a user protests that 5 tablespoons of tabasco sauce is legitimate… if you’re making a cake to prank a friend.

    - In that case, you could add a wrinkle to the policy of “... unless the cake is known to be made as a prank”.

    - If there are wrinkles that a small but consistent set of independent savvy users want, you might be able to expand that logic to the general populace, getting a self-wrinkling set of default policies that handle most cases well.

    - This is less like a top-down ontology, more like an emergent folksonomy that can grow itself by starting from a good-enough crystallized background knowledge.

  - Now, instead of hyper-precise clockwork gears, you have rough clay that you can smoosh into place.

    - More organic than mechanical.

- The MidJourney-style 4-up choices is a powerful UI paradigm.

  - For image generation, it’s the best way to steer the generation through the latent space.

  - But it actually makes sense as an iteration style for *any* movement through a state space.

  - For example, imagine a code artifact, and a series of diffs to modify it.

  - A knowledgeable user might be able to create a bespoke diff that matches their intuited direction of travel.

  - But a less knowledgeable user might want to tell an assistive technology what their intent was, see multiple possible diffs, and then pick the one that seems most in line with their intent.

  - A 4-up is also much more forgiving for lower quality.

    - As long as a good enough answer exists in one of the four options, then the user is satisfied (at least to sit through another round of iteration).

    - If you only had one option, the likelihood the option was good enough is significantly lower.

    - Imagine the likelihood of a good-enough-quality generation is 70%.

    - For a single one, the chance it’s bad enough quality to get the user to give up is 30%.

    - For a 4 up, it’s 0.8% (0.30 ^ 4)

  - The 4-up also allows giving more variation, which allows the user to lean into the one that feels most right to them, giving feedback that might be hard to articulate in precise terms.

- A key question for a UI: how expensive is the tool when it’s wrong?

  - That is, when its suggestion is below the good-enough bar.

  - If it’s a primary use case and a one-shot answer with no recourse to generate another one, it could be extremely expensive.

    - The user had to:

      - Think to use the tool

      - Launch the tool

      - Give it enough context on their goal

      - Wait for the answer

      - Evaluate the answer

    - That’s a lot of wasted time if it ends up not being good enough!

  - But imagine it’s a secondary use case.

  - You came to the tool for another primary use case, which works dependly.

  - Off to the side, you see a suggestion, perhaps a 4-up.

    - If one of the suggestions is good, you have a magical experience.

    - If the suggestions are not good, but they’re easy to ignore or skim, then the cost is miniscule, just the flick of your eyes there and back.

  - This makes secondary use cases much more forgiving for low quality.

- The security notion of “untrusted” is confusingly named.

  - A layperson might view “untrusted” as a bad thing.

  - But actually it’s a good thing!

  - What it really means is “a thing that you don’t *have* to trust”, which is good!

  - The more components that can be untrusted, the more rigorously the system works!

- Running arbitrary compositions of code in a highly locked down sandbox is easy.

  - If code inside an impenetrable sandbox does something dangerous, does it matter?

    - If a tree falls in the woods and there’s no one around to hear it, does it make a sound?

  - But for the computation to do anything *useful*, it has to interact with the surrounding world.

    - For example, reaching out to the merchant’s server to let them know the user has requested to buy the product.

  - That requires that *sometimes* the data must escape.

    - This is the hard part!

    - That “sometimes” is an absolute beast of a problem!

- LLMs mean that anything with an API can now be controlled in plain english.

  - There are a ton of *amazing* open source tools and frameworks that previously were a bit fiddly to use.

    - Some powerful libraries, like ThreeBlueOneBrown’s [<u>manim</u>](https://github.com/3b1b/manim), allow amazing animations, but can be hard to use if you aren’t comfortable programming.

    - Lots of open-source tools that have a graphical user interface are confusing to use.

      - Open source tools with a GUI tend to have confusing user experiences, because a coherent UX requires a coherent vision, and swarms of open source volunteer effort can't do that easily.

  - But now, LLMs can help you use all of the power of these tools as long as they have an API.

  - A friend was able to use a hosted notebook running manim to take an english-language description of a visualization and, with a few automatic auto syntax-error fixing cycles, have a final rendered video.

- The whole point of an ecosystem is open endedness.

  - The safer the composition of untrusted components, the more open-ended the system: the larger the combinatorial possibility.

- Written artifacts are landmarks to navigate within a given idea maze.

  - (This is riffing off an observation from Gordon Brander.)

  - When we’re feeling our way through the idea maze, we hold threads of analysis lightly; a living, evolving, fragile hypothesis.

  - The more that they seem to help us, the more that they survive disconfirming evidence, the more that they resonate with different people, the more confident we become that the thread is onto something useful.

  - Once you get confident enough that the idea is durable and useful, you want to freeze it in time; to memorialize it in a way that you can point others to.

  - This process of crystallizing a living thought into a frozen artifact takes time and effort, but it’s worth it when you know it’s useful.

  - By doing this effort you can now transmit the idea, for ~free, into the future and into a significantly larger number of heads much more cheaply.

  - The document becomes a kind of tombstone for the original thread of analysis; locking it in in a more durable way.

  - Those tombstones then become landmarks to help you and others navigate that particular idea maze.

- I learned of a new concept: positive deviance.

  - I learned about it via Aishwarya Khanduja’s phenomenal blog: [<u>https://www.aishwaryadoingthings.com/the-whole-is-greater-than-the-sum-of-its-parts</u>](https://www.aishwaryadoingthings.com/the-whole-is-greater-than-the-sum-of-its-parts)

    - Another great article: [<u>https://www.aishwaryadoingthings.com/from-physics-envy-to-biology-envy</u>](https://www.aishwaryadoingthings.com/from-physics-envy-to-biology-envy)

  - From Wikipedia’s article:

    - Positive deviance (PD) is an approach to behavioral and social change. It is based on the idea that, within a community, some individuals engage in unusual behaviors allowing them to solve problems better than others who face similar challenges, despite not having additional resources or knowledge. These individuals are referred to as positive deviants

  - Positive deviance is positive noise in a system.

  - A system that has lots of positive deviance within it is antifragile: it will get stronger from stress.

- When someone is multiple ply behind you or not seeing all of the relevant dimensions they might say “I’m happy to believe that… but first prove to me that water is wet”.

  - Those discussions take *forever* and are pushing a rock up hill that might roll back at any time.

  - The only way to avoid them is collaborators who already have the knowhow to sense the dimension, and no higher-ranking people you need to convince who don’t have the necessary knowhow.

  - If a junior person doesn't get it but can execute anyway that's fine.

  - If a person who has the authority to block you doesn't get it, it can be a serious problem.

- When you work at multiple organizations, you can factor out the kayfabe to surf it.

  - Kayfabe within an organization is like gravity: omnipresent.

    - You just take it for granted as always working that way.

    - It blends into the background, something that you can’t see even when looking directly at it.

  - But the specific flavor of kayfabe within a given organization will differ.

  - This means that if you spend all of your time in one organizational environment you might overfit to it.

    - Your intuitions get increasingly poorly suited to other domains.

  - But if you’ve spent time in two organizations (or ideally more), the kayfabe pops out to you from the difference, almost like a magic eye picture.

  - Once you can see it, you can now navigate it, and surf it.

    - Instead of it jostling you around, you can ride it towards better outcomes.

  - Another example of where having one foot inside a system and one foot outside allows you leverage.

- There’s a big difference between reversible and irreversible errors.

  - Irreversible errors can be game over: you have to be conservative.

  - Reversible errors are useful noise to grow and learn from.

    - See the error as a happy little accident that makes you stronger.

    - The error helps you discover the boundaries.

  - Systems with a higher proportion of reversible errors will be stronger and more anti-fragile.

  - This is one of the reasons that high-trust organizations can create great results!

- Adversarial collaborations will share the bare minimum.

  - Adversarial collaborations happen when the two collaborators don’t trust each other.

  - They share the bare minimum, just to be safe.

  - This creates a very thin thread of information flow.

    - This makes it unlikely to have unexpected downside…

    - … but also make it very unlikely to have unexpected upside.

  - This is one of the reasons that high-trust contexts can discover great ideas.

    - They’re default collaborative, happy to share information.

    - And the more information that is shared, the more likely you are to find a great new combination in it, or discover relevant disconfirming evidence.

  - You can make some contexts higher-trust by having a rigorous system in place.

    - For example, societies with rigorous contract law and enforcement are significantly more innovative.

    - Participants can default-trust each other and collaborate in a positive-sum way.

  - If you have the right laws of physics to allow a default-collaborative system, it can innovate significantly faster than other systems.

- As humanity we lost agency over our tools.

  - The way we'll regain it is by changing the laws of physics.

- Product designers typically assume users are fundamentally lazy.

  - What if you assume users are fundamentally creative?

- Effective jargon is an import statement for mutual understanding among specific collaborators in a given context.

- If you know you're playing with a certified 10-dimensional chess player, how can you possibly trust them?

  - They could be totally taking advantage of you in the dimensions beyond the ones you grok.

  - In the novel *Blindsight*, there are vampires who are phenomenal at intuitively grokking complexity... but they're extremely dangerous!

- Things with massive network effects are inherently wildly path dependent.

  - Small starting noise blown up to galactic proportions.

  - The slight bumpiness in the distribution of matter after the big bang determined where whole galaxies would form.

- In the past I’ve talked about the [<u>nerd club</u>](#s0cfteif5ebc) pattern.

  - It’s an optional/secret group that allows a diverse group of people to choose to talk about whatever they want.

  - This means that the threads that people choose to participate in are ones they find interesting.

    - Interesting: surprising in some way, and potentially useful.

    - Interesting things spark people’s curiosity.

  - A thread that many different people find interesting is likely to be found interesting in the surrounding context, too.

    - This is not the case if you have an echo chamber of similar people.

    - But because you’ve added people to the club with a “novelty search” over different viewpoints, it gives you a good proxy for how the overall context will find it.

  - This is why nerd clubs are structurally more likely to find interesting, game-changing ideas.

- An elegant idea blooms in your brain fractally.

  - Physicists like hyper compact distillations that bloom and unfurl fractally in your mind as you consider them more deeply.

  - To people who aren't curious or don't have the necessary background knowledge, the elegant distillation will look perplexing, opaque, unrelated to the question they asked.

  - But when you have the ability and time to make them bloom, they are transcendent.

- When designing an emergent design system, have the smallest alphabet but no limits on combinations.

  - Add to the alphabet extremely carefully, the smallest addition that enables the most combinatorial possibility.

    - Conceptually like a minimum spanning tree.

  - It’s way easier to add than to remove, and the more characters you add to the alphabet, the harder it will be to reign in.

- Every context becomes a bit of an echo chamber.

  - It gets more and more set in its ways, more and more optimized for the patterns that have been known to work.

  - This is when an environment might become “old money”.

  - If something new comes along that breaks the mental models, it simply won’t fit!

  - The Silicon Valley tech scene is heavily optimized for this late stage vertical Saas world.

- “Late stage” means when the system has gotten hyper optimized to the point of brittleness.

  - Absurd outcomes.

  - Flattening nuance into a small number of legible dimensions.

  - Chase is [<u>shipping a bespoke advertising network within their app</u>](https://www.pymnts.com/connectedeconomy/2024/chase-shows-super-app-ambitions-with-media-platform-launch/). How late stage!

- Deciding to join an ecosystem as a creator is about the short term benefit vs the long term downside.

  - Once creators join into an aggregator it's very hard for them to leave. But they can decide not to participate.

  - The short term benefit might be marginal user traffic or engagement to your content.

  - The long term downside might be tying yourself to an entity that will have increasing power or control over you.

    - Submitting yourself to a feudal lord that grows more powerful every day.

    - That feudal lord may change the rules later in the game in a way that significantly harms the creators; almost every aggregator in history has!

  - Only if the short-term benefit beats out the long-term downside will the marginal participant decide to participate.

    - But of course the short-term benefit is often concrete and clear and the long-term downside is often diffuse and abstract.

    - This gives an edge to the short-term benefit winning out.

  - However, when the marginal entity is deciding to participate, if they have two options, both with roughly equivalent short term benefit but one with significantly lower downside risk, they will have a distinct edge to the one with the lower downside risk.

    - This can often be a small but distinct asymmetry, an edge.

    - Ecosystems have significant network effects, which means a consistent edge (even a small one!) can rapidly accumulate an overwhelming advantage.

    - A small but consistent and distinct asymmetry with a network effect can become a hurricane.

  - This is why if there's an open system that is roughly equivalent in short-term benefit to the proto-aggregator, the open system will win.

- Why doesn't every ecosystem just do the aggregator pattern?

  - (Ignoring the fact for a moment that I think the aggregator pattern is often less beneficial for society overall.)

  - The reason is because it's impossible to become an aggregator from a standstill.

    - You can't simply create a platform and have the aggregator pattern develop.

  - You first need significant, even dominant, consumer activity in that category, which you can *then* develop into an aggregator by bending the incentives of content producers.

    - You need to get the lines of gravity to loop back into yourself and get the aggregator flywheel going.

    - This requires a mass that stands out prominently from everything else around it.

  - A proto-aggregator will also have significantly worse potential of downside risk for marginal creators, so if they have no short-term benefit to offer creators, they will never get going.

  - If you already have massive consumer engagement that is significantly beyond competitors, you can parlay it into becoming an aggregator.

  - But if you don’t, your best choice is to open up your ecosystem to make it clear that you can’t become an aggregator, so you have an edge over other similarly nascent ecosystems.

- How do mob boss dynamics work?

  - Nearly every entity that interacts with them is terrified of the mob boss, but none of them stand up or even say a critical thing, often even in private. Why?

  - Because they know that the boss has tremendous power over them, and also that the boss is eager to capriciously use that power, including in private ways that no one else can see.

    - It’s always hearsay and gossip, whispered warnings of over-the-top and terrifying things that may or may not have happened, chilling everyone’s confidence.

  - It's a collective action problem.

    - If everyone stood up to the mob boss, they'd be toppled.

    - But if anyone defects, then whoever sticks their neck out will get punished.

    - Individuals can't stand up or they'll be crushed.

    - So everyone goes along with it, pretending to be OK with it, but desperately hoping for a change.

  - A supercritical state.

    - Looks extremely solid and unchangeable, but ready for the right inciting incident and a landslide can change everything.

  - These kinds of dynamics are typical in late-stage ecosystems with highly powerful central gatekeepers.

- I’m not a fan of Twitter for nuanced discourse.

  - Nuance takes time and space to establish.

  - When you have to communicate your beliefs quickly and in little space to people who don’t share context, they become caricatured, high-contrast.

  - This is especially true when you have a large number of people watching; every conversation becomes automatically charged.

  - Real beliefs are almost always more nuanced than the viral, fast-to-consume derivations of them.

  - That’s why I prefer cozy environments for rigorous, nuanced collaborative debate.

  - If you engage me in a nuanced discussion on Twitter, I’m likely to not engage in that forum.

    - However, I’d be happy to engage deeply in a more cozy setting.

- Everything at the quantum mechanics level is overwhelmingly noisy.

  - If you try to keep track of every detail, your brain will explode.

  - But if you average everything out at the right scale, it turns out a number of stable, clear dynamics pop out.

  - A [<u>great video on YouTube</u>](https://www.youtube.com/watch?v=MXs_vkc8hpY) on fluid dynamics at multiple levels of abstraction that made this very intuitive for me.

  - My time on search at Google had a similar vibe, I realize.

  - “All of the random human behavior on the web is weird noise… but if you average it correctly, you can get extremely clean, useful signal!”

    - One of the reasons this works is that no individual instance of a human decision (e.g. what query to issue, which pages choose to link to which others) actually changes the emergent outcome of search ranking much, which reduces the desire to game them.

    - But in *aggregate*, those signals are enormously valuable, a summarized human intent.

- An article I found interesting: [<u>We Need To Rewild The Internet</u>](https://www.noemamag.com/we-need-to-rewild-the-internet/)

  - “The internet has become an extractive and fragile monoculture. But we can revitalize it using lessons learned by ecologists.”

- Ben Thompson’s [<u>MKBHDs For Everything</u>](https://stratechery.com/2024/mkbhds-for-everything/) was excellent.

  - “The connection between us and AI, though, is precisely the fact that we haven’t needed it: the nature of media is such that we could already create text and video on our own, and take advantage of the Internet to — at least in the case of Brownlee — deliver finishing blows to \$230 million startups.

  - How many industries, though, are not media, in that they still need a team to implement the vision of one person? How many apps or services are there that haven’t been built, not because one person can’t imagine them or create them in their mind, but because they haven’t had the resources or team or coordination capabilities to actually ship them?

  - This gets at the vector through which AI impacts the world above and beyond cost savings in customer support, or whatever other obvious low-hanging fruit there may be: as the ability of large language models to understand and execute complex commands — with deterministic computing as needed — increases, so too does the potential power of the sovereign individual telling AI what to do. The Internet removed the necessity — and inherent defensibility — of complex cost structures for media; AI has the potential to do the same for a far greater host of industries.